mod network;
mod vec3;

use network::{Layer, Matrix, Network};
use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use serde_json::Value;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use vec3::Vec3;

use std::f32::consts::PI;

fn load_tensor(path: &Path, dims: &[usize]) -> Vec<f32> {
    let bytes = fs::read(path).expect("read tensor");
    let mut scalars = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        scalars.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    scalars
}

fn load_shapes(path: &Path) -> Vec<(String, Vec<usize>)> {
    fs::read_to_string(path)
        .expect("read shapes")
        .lines()
        .map(|line| {
            let mut parts = line.split_whitespace();
            let name = parts.next().unwrap().to_string();
            let dims = parts.map(|p| p.parse().unwrap()).collect();
            (name, dims)
        })
        .collect()
}

fn load_tf_samples(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let samples: Value = serde_json::from_str(&content)?;
    Ok(samples)
}

//fn compute_final_color(o: Vec3, d: Vec3, t: &Vec<f32>, network: &Network) -> Vec3 {
//    let mut c = Vec::new();
//    let mut sigma = Vec::new();
//    let n = t.len();
//    let d_hat = d.normalize();

//    for i in 0..n {
//        let p = o + d_hat * t[i];
//        let (c_i, sigma_i) = network.forward(&p, &d_hat);
//        c.push(c_i);
//        sigma.push(sigma_i);
//    }
//    let mut t_prod = 1.0f32;
//    let mut c_final = Vec3::new(0.0, 0.0, 0.0);
//    for i in 0..n - 1 {
//        let delta_i = t[i + 1] - t[i];
//        let alpha_i = 1.0 - (-sigma[i] * delta_i).exp();
//        let w_i = t_prod * alpha_i;
//        c_final += c[i] * w_i;

//        t_prod *= 1.0 - alpha_i;
//    }
//    c_final
//}

fn compute_final_color(o: Vec3, d: Vec3, t: &[f32], network: &Network, far: f32) -> Vec3 {
    let d_hat = d.normalize();
    let n = t.len();

    let mut T = 1.0f32;                // transmittance
    let mut rgb = Vec3::new(0.0, 0.0, 0.0);
    let mut acc = 0.0f32;

    for i in 0..n {
        let p = o + d_hat * t[i];
        let (c_i, sigma_i) = network.forward(&p, &d_hat);

        // Use last interval out to 'far' (NeRF uses a very large last delta)
        let delta = if i + 1 < n { t[i + 1] - t[i] } else { far - t[i] };

        let alpha = 1.0 - (-sigma_i * delta).exp();
        let w = T * alpha;

        rgb += c_i * w;
        acc += w;
        T *= 1.0 - alpha;

        if T < 1e-4 { break; } // optional early-out
    }

    rgb += Vec3::new(1.0, 1.0, 1.0) * (1.0 - acc);
    rgb
}


struct Camera {
    nx: usize,
    ny: usize,

    alpha_width: f32,
    alpha_height: f32,

    pos: Vec3,
    dir: Vec3,
    up: Vec3,

    near: f32,
    far: f32,
    samples_per_ray: usize,
}

impl Camera {
    fn get_ray_dir(&self, i: usize, j: usize) -> Vec3 {
        // Orthonormal basis
        let f = self.dir.normalize();
        let r = f.cross(&self.up).normalize(); // right
        let u = r.cross(&f).normalize();       // true up

        // Pixel center in NDC [-1,1] with y up
        let x = ( (j as f32 + 0.5) / self.nx as f32 ) * 2.0 - 1.0;
        let y = 1.0 - ( (i as f32 + 0.5) / self.ny as f32 ) * 2.0;

        // FOV to slopes
        let sx = self.alpha_width.tan();
        let sy = self.alpha_height.tan();

        // Build & return (normalized later)
        r * (x * sx) + u * (y * sy) + f
    }
}

fn get_sample_locs(near: f32, far: f32, n: usize) -> Vec<f32> {
    let mut t = Vec::with_capacity(n);
    let step = (far - near) / n as f32;
    for i in 0..n {
        let u = 0.5;
        t.push(near + (i as f32 + u) * step);
    }
    t
}

fn render_image(network: &Network, camera: &Camera) -> Vec<Vec3> {
    let o = camera.pos;
    let total_pixels = camera.nx * camera.ny;
    let pixel_count = Arc::new(AtomicUsize::new(0));
    let pixel_count_clone = pixel_count.clone();
    
    // Create a vector of pixel indices to process in parallel
    let pixel_indices: Vec<(usize, usize)> = (0..camera.ny)
        .flat_map(|i| (0..camera.nx).map(move |j| (i, j)))
        .collect();
    
    let image: Vec<Vec3> = pixel_indices
        .par_iter()
        .map(|&(i, j)| {
            let d = camera.get_ray_dir(i, j);
            let t = get_sample_locs(camera.near, camera.far, camera.samples_per_ray);
            let c = compute_final_color(o, d, &t, network, camera.far);
            
            let count = pixel_count_clone.fetch_add(1, Ordering::Relaxed) + 1;
            if count % 5000 == 0 {
                let progress = (count as f32 / total_pixels as f32) * 100.0;
                println!("Rendering progress: {}/{} pixels ({:.1}%)", count, total_pixels, progress);
            }
            
            c
        })
        .collect();
    
    println!("Rendering complete: {}/{} pixels (100.0%)", total_pixels, total_pixels);
    image
}

fn save_ppm(path: &Path, width: usize, height: usize, pixels: &[Vec3]) -> std::io::Result<()> {
    assert_eq!(pixels.len(), width * height);
    let mut f = BufWriter::new(File::create(path)?);
    write!(f, "P6\n{} {}\n255\n", width, height)?;
    let mut buf = Vec::with_capacity(pixels.len() * 3);
    for p in pixels {
        let r = (p.x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let g = (p.y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let b = (p.z.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        buf.extend_from_slice(&[r, g, b]);
    }
    f.write_all(&buf)
}

fn get_vec3(node: &Value, key: &str) -> Vec3 {
    let arr = node[key].as_array().expect(key);
    Vec3::new(
        arr[0].as_f64().unwrap() as f32,
        arr[1].as_f64().unwrap() as f32,
        arr[2].as_f64().unwrap() as f32,
    )
}

fn main() {
    let root = Path::new("/Users/elisabeth/projects/nerf-rs/lego_rust/");

    let mut dense0_kernel = Matrix::empty();
    let mut dense0_bias = Vec::new();
    let mut dense1_kernel = Matrix::empty();
    let mut dense1_bias = Vec::new();
    let mut dense2_kernel = Matrix::empty();
    let mut dense2_bias = Vec::new();
    let mut dense3_kernel = Matrix::empty();
    let mut dense3_bias = Vec::new();
    let mut dense4_kernel = Matrix::empty();
    let mut dense4_bias = Vec::new();
    let mut dense5_kernel = Matrix::empty();
    let mut dense5_bias = Vec::new();
    let mut dense6_kernel = Matrix::empty();
    let mut dense6_bias = Vec::new();
    let mut dense7_kernel = Matrix::empty();
    let mut dense7_bias = Vec::new();
    let mut bottleneck_kernel = Matrix::empty();
    let mut bottleneck_bias = Vec::new();
    let mut viewdirs_kernel = Matrix::empty();
    let mut viewdirs_bias = Vec::new();
    let mut rgb_kernel = Matrix::empty();
    let mut rgb_bias = Vec::new();
    let mut alpha_kernel = Matrix::empty();
    let mut alpha_bias = Vec::new();

    for (name, dims) in load_shapes(&root.join("coarse/shapes.txt")) {
        let data = load_tensor(&root.join(format!("coarse/{name}.bin")), &dims);
        match name.as_str() {
            "dense0_kernel" => dense0_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "dense0_bias" => dense0_bias = data,
            "dense1_kernel" => dense1_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "dense1_bias" => dense1_bias = data,
            "dense2_kernel" => dense2_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "dense2_bias" => dense2_bias = data,
            "dense3_kernel" => dense3_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "dense3_bias" => dense3_bias = data,
            "dense4_kernel" => dense4_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "dense4_bias" => dense4_bias = data,
            "dense5_kernel" => dense5_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "dense5_bias" => dense5_bias = data,
            "dense6_kernel" => dense6_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "dense6_bias" => dense6_bias = data,
            "dense7_kernel" => dense7_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "dense7_bias" => dense7_bias = data,
            "bottleneck_kernel" => {
                bottleneck_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32)
            }
            "bottleneck_bias" => bottleneck_bias = data,
            "viewdirs_kernel" => {
                viewdirs_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32)
            }
            "viewdirs_bias" => viewdirs_bias = data,
            "rgb_kernel" => rgb_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "rgb_bias" => rgb_bias = data,
            "alpha_kernel" => alpha_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "alpha_bias" => alpha_bias = data,
            _ => panic!("Unknown tensor: {}", name),
        }
        //println!("{} has {} values", name, data.len());
    }
    let layers = vec![
        Layer::new(dense0_kernel, dense0_bias),
        Layer::new(dense1_kernel, dense1_bias),
        Layer::new(dense2_kernel, dense2_bias),
        Layer::new(dense3_kernel, dense3_bias),
        Layer::new(dense4_kernel, dense4_bias),
        Layer::new(dense5_kernel, dense5_bias),
        Layer::new(dense6_kernel, dense6_bias),
        Layer::new(dense7_kernel, dense7_bias),
    ];
    let bottleneck = Layer::new(bottleneck_kernel, bottleneck_bias);
    let viewdirs = Layer::new(viewdirs_kernel, viewdirs_bias);
    let rgb = Layer::new(rgb_kernel, rgb_bias);
    let alpha = Layer::new(alpha_kernel, alpha_bias);

    let network = Network::new(layers, bottleneck, viewdirs, rgb, alpha);

    let samples = load_tf_samples(&root.join("tf_reference_samples.json")).unwrap();

    let near = samples["near"].as_f64().unwrap() as f32;
    let far = samples["far"].as_f64().unwrap() as f32;
    let samples_per_ray = samples["samples_per_ray"].as_u64().unwrap() as usize;
    let pos = get_vec3(&samples, "camera_origin");
    let dir = get_vec3(&samples, "camera_forward").normalize();
    let up = get_vec3(&samples, "camera_up").normalize();

    // print samples per ray
    println!("Samples per ray: {}", samples_per_ray);

    // fill in image resolution/FOV however you like
    let cam = Camera {
        nx: 512,
        ny: 512,
        alpha_width: PI / 8.0,
        alpha_height: PI / 8.0,
        pos,
        dir,
        up,
        near,
        far,
        samples_per_ray
    };

    println!("Starting image rendering...");
    let render_start = Instant::now();
    let image = render_image(&network, &cam);
    let render_duration = render_start.elapsed();
    
    println!("Rendering completed in {:.2} seconds", render_duration.as_secs_f64());
    save_ppm(&Path::new("output.ppm"), cam.nx, cam.ny, &image).unwrap();
}
