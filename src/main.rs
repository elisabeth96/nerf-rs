mod network;
mod vec3;

use network::{Layer, Matrix, Network};
use rayon::prelude::*;
use serde_json::Value;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use vec3::Vec3;

use std::f32::consts::PI;

fn load_tensor(path: &Path, _dims: &[usize]) -> Vec<f32> {
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

fn integrate_ray(colors: &[Vec3], sigmas: &[f32], t: &[f32], far: f32) -> Vec3 {
    let n = t.len();
    if n == 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    debug_assert_eq!(colors.len(), n);
    debug_assert_eq!(sigmas.len(), n);

    let mut transmittance = 1.0f32;
    let mut rgb = Vec3::new(0.0, 0.0, 0.0);
    let mut acc = 0.0f32;

    for i in 0..n {
        let c_i = colors[i];
        let sigma_i = sigmas[i];

        let delta = if i + 1 < n {
            t[i + 1] - t[i]
        } else {
            far - t[i]
        };

        let alpha = 1.0 - (-sigma_i * delta).exp();
        let w = transmittance * alpha;

        rgb += c_i * w;
        acc += w;
        transmittance *= 1.0 - alpha;

        if transmittance < 1e-4 {
            break;
        }
    }

    rgb + Vec3::new(1.0, 1.0, 1.0) * (1.0 - acc)
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
        let u = r.cross(&f).normalize(); // true up

        // Pixel center in NDC [-1,1] with y up
        let x = ((j as f32 + 0.5) / self.nx as f32) * 2.0 - 1.0;
        let y = 1.0 - ((i as f32 + 0.5) / self.ny as f32) * 2.0;

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
    let origin = camera.pos;
    let total_pixels = camera.nx * camera.ny;
    let sample_positions = get_sample_locs(camera.near, camera.far, camera.samples_per_ray);
    let pixel_count = Arc::new(AtomicUsize::new(0));
    let pixel_count_clone = pixel_count.clone();

    let block_size = 8usize;
    let block_coords: Vec<(usize, usize)> = (0..camera.ny)
        .step_by(block_size)
        .flat_map(|block_y| {
            (0..camera.nx)
                .step_by(block_size)
                .map(move |block_x| (block_y, block_x))
        })
        .collect();

    let block_results: Vec<Vec<(usize, Vec3)>> = block_coords
        .par_iter()
        .map(|&(block_y, block_x)| {
            let mut rays = Vec::new();
            let max_y = (block_y + block_size).min(camera.ny);
            let max_x = (block_x + block_size).min(camera.nx);
            for i in block_y..max_y {
                for j in block_x..max_x {
                    let dir = camera.get_ray_dir(i, j);
                    rays.push((i * camera.nx + j, dir.normalize()));
                }
            }

            if rays.is_empty() {
                return Vec::new();
            }

            let samples_per_ray = sample_positions.len();
            let mut results = Vec::with_capacity(rays.len());

            if samples_per_ray == 0 {
                for (pixel_index, _) in &rays {
                    results.push((*pixel_index, Vec3::new(0.0, 0.0, 0.0)));
                }
            } else {
                let total_samples = rays.len() * samples_per_ray;
                let mut points = Matrix::zeros(3, total_samples as i32);
                let mut view_dirs = Vec::with_capacity(total_samples);

                for (ray_idx, (_, dir_hat)) in rays.iter().enumerate() {
                    for (sample_idx, &ti) in sample_positions.iter().enumerate() {
                        let col = ray_idx * samples_per_ray + sample_idx;
                        let p = origin + *dir_hat * ti;
                        points.set(0, col, p.x);
                        points.set(1, col, p.y);
                        points.set(2, col, p.z);
                        view_dirs.push(*dir_hat);
                    }
                }

                let (sample_colors, sample_sigmas) = network.forward_batch(&points, &view_dirs);

                for (ray_idx, (pixel_index, _)) in rays.iter().enumerate() {
                    let start = ray_idx * samples_per_ray;
                    let end = start + samples_per_ray;
                    let color = integrate_ray(
                        &sample_colors[start..end],
                        &sample_sigmas[start..end],
                        &sample_positions,
                        camera.far,
                    );
                    results.push((*pixel_index, color));
                }
            }

            let processed = pixel_count_clone.fetch_add(rays.len(), Ordering::Relaxed) + rays.len();
            let prev = processed - rays.len();
            if processed / 5000 != prev / 5000 {
                let progress = (processed as f32 / total_pixels as f32) * 100.0;
                println!(
                    "Rendering progress: {}/{} pixels ({:.1}%)",
                    processed, total_pixels, progress
                );
            }

            results
        })
        .collect();

    let mut image = vec![Vec3::new(0.0, 0.0, 0.0); total_pixels];
    for block in block_results {
        for (index, color) in block {
            image[index] = color;
        }
    }

    println!(
        "Rendering complete: {}/{} pixels (100.0%)",
        total_pixels, total_pixels
    );

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
        samples_per_ray,
    };

    println!("Starting image rendering...");
    let render_start = Instant::now();
    let image = render_image(&network, &cam);
    let render_duration = render_start.elapsed();

    println!(
        "Rendering completed in {:.2} seconds",
        render_duration.as_secs_f64()
    );
    save_ppm(&Path::new("output.ppm"), cam.nx, cam.ny, &image).unwrap();
}
