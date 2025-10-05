mod network;
mod vec3;

use std::fs;
use serde_json::Value;
use std::path::Path;
use rand::Rng;
use network::{Matrix, Layer, Network};
use vec3::Vec3;

fn load_tensor(path: &Path, dims: &[usize]) -> Vec<f32> {
    let bytes = fs::read(path).expect("read tensor");
    let mut scalars = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        scalars.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    //assert_eq!(scalars.len(), dims.iter().product());
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

fn compute_final_color (o: Vec3, d: Vec3, near : f32, far : f32, samples_per_ray : usize, network: &Network) -> Vec3 {
    // Create a random number generator
    let mut rng = rand::thread_rng();
    let mut t = Vec::new();
    let mut c = Vec::new();
    let mut sigma = Vec::new();

    for i in 0..samples_per_ray {
        let xi: f32 = rng.gen_range(0.0..1.0);
        let t_i = near + (i as f32 + xi) / samples_per_ray as f32 * (far - near);
        let p = o + d.normalize() * t_i;
        let (c_i, sigma_i) = network.forward(&p, &d.normalize());
        t.push(t_i);
        c.push(c_i);
        sigma.push(sigma_i);
    }
    let mut t_prod = 1.0f32;
    let mut c_final = Vec3::new(0.0, 0.0, 0.0);
    for i in 0..samples_per_ray-1 {
        let delta_i = t[i+1] - t[i];
        let alpha_i = 1.0 - (-sigma[i] * delta_i).exp();
        let w_i = t_prod * alpha_i;
        c_final += c[i] * w_i;

        t_prod *= 1.0 - alpha_i;
    }
    c_final
}

fn main() {
    let root = Path::new("/Users/elisabeth/projects/nerf-rs/lego_rust/");

    match load_tf_samples(&root.join("tf_reference_samples.json")) {
        Ok(samples) => {
            println!("JSON structure: {}", serde_json::to_string_pretty(&samples).unwrap());
            
            // Extract values from the first sample
            if let Some(examples) = samples["examples"].as_array() {
                if !examples.is_empty() {
                    let first_sample = &examples[0];
                    println!("\nFirst sample data:");
                    
                    // Extract pixel coordinates
                    if let Some(pixel) = first_sample["pixel"].as_array() {
                        if pixel.len() >= 2 {
                            let x = pixel[0].as_u64().unwrap();
                            let y = pixel[1].as_u64().unwrap();
                            println!("  Pixel: ({}, {})", x, y);
                        }
                    }
                    
                    // Extract ray origin
                    if let Some(ray_o) = first_sample["ray_o"].as_array() {
                        if ray_o.len() >= 3 {
                            let x = ray_o[0].as_f64().unwrap();
                            let y = ray_o[1].as_f64().unwrap();
                            let z = ray_o[2].as_f64().unwrap();
                            println!("  Ray origin: ({}, {}, {})", x, y, z);
                        }
                    }
                    
                    // Extract ray direction
                    if let Some(ray_d) = first_sample["ray_d"].as_array() {
                        if ray_d.len() >= 3 {
                            let x = ray_d[0].as_f64().unwrap();
                            let y = ray_d[1].as_f64().unwrap();
                            let z = ray_d[2].as_f64().unwrap();
                            println!("  Ray direction: ({}, {}, {})", x, y, z);
                        }
                    }
                    
                    // Extract coarse_rgb values (all of them)
                    if let Some(coarse_rgb) = first_sample["coarse_rgb"].as_array() {
                        println!("  Coarse RGB values:");
                        for (i, rgb_value) in coarse_rgb.iter().enumerate() {
                            if let Some(rgb_array) = rgb_value.as_array() {
                                if rgb_array.len() >= 3 {
                                    let r = rgb_array[0].as_f64().unwrap();
                                    let g = rgb_array[1].as_f64().unwrap();
                                    let b = rgb_array[2].as_f64().unwrap();
                                    println!("    Sample {}: ({}, {}, {})", i, r, g, b);
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }
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
            "bottleneck_kernel" => bottleneck_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "bottleneck_bias" => bottleneck_bias = data,
            "viewdirs_kernel" => viewdirs_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "viewdirs_bias" => viewdirs_bias = data,
            "rgb_kernel" => rgb_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "rgb_bias" => rgb_bias = data,
            "alpha_kernel" => alpha_kernel = Matrix::new(data, dims[0] as i32, dims[1] as i32),
            "alpha_bias" => alpha_bias = data,
            _ => panic!("Unknown tensor: {}", name),
        }
        //println!("{} has {} values", name, data.len());
    }
    let layers = vec![Layer::new(dense0_kernel, dense0_bias), Layer::new(dense1_kernel, dense1_bias), Layer::new(dense2_kernel, dense2_bias), Layer::new(dense3_kernel, dense3_bias), Layer::new(dense4_kernel, dense4_bias), Layer::new(dense5_kernel, dense5_bias), Layer::new(dense6_kernel, dense6_bias), Layer::new(dense7_kernel, dense7_bias)];
    let bottleneck = Layer::new(bottleneck_kernel, bottleneck_bias);
    let viewdirs = Layer::new(viewdirs_kernel, viewdirs_bias);
    let rgb = Layer::new(rgb_kernel, rgb_bias);
    let alpha = Layer::new(alpha_kernel, alpha_bias);

    let network = Network::new(layers, bottleneck, viewdirs, rgb, alpha);

    // load from json
    let samples = load_tf_samples(&root.join("tf_reference_samples.json")).expect("load tf reference samples");
    let examples = samples["examples"].as_array().expect("examples array");
    let first = &examples[0];

    let ray_o = first["ray_o"].as_array().expect("ray_o array");
    let ray_d = first["ray_d"].as_array().expect("ray_d array");
    let o = Vec3::new(
        ray_o[0].as_f64().unwrap() as f32,
        ray_o[1].as_f64().unwrap() as f32,
        ray_o[2].as_f64().unwrap() as f32,
    );
    let d = Vec3::new(
        ray_d[0].as_f64().unwrap() as f32,
        ray_d[1].as_f64().unwrap() as f32,
        ray_d[2].as_f64().unwrap() as f32,
    );

    let near = samples["near"].as_f64().unwrap() as f32;
    let far = samples["far"].as_f64().unwrap() as f32;
    let samples_per_ray = samples["samples_per_ray"].as_u64().unwrap() as usize;

    // compute expected color from reference coarse outputs using provided z_vals
    let z_vals: Vec<f32> = samples["z_vals"]
        .as_array()
        .expect("z_vals array")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let coarse_sigma: Vec<f32> = first["coarse_sigma"]
        .as_array()
        .expect("coarse_sigma array")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let coarse_rgb_vals = first["coarse_rgb"].as_array().expect("coarse_rgb array");
    let coarse_rgb: Vec<Vec3> = coarse_rgb_vals
        .iter()
        .map(|rgb| {
            let a = rgb.as_array().unwrap();
            Vec3::new(a[0].as_f64().unwrap() as f32, a[1].as_f64().unwrap() as f32, a[2].as_f64().unwrap() as f32)
        })
        .collect();

    let mut transmittance = 1.0f32;
    let mut c_expected = Vec3::new(0.0, 0.0, 0.0);
    for i in 0..(z_vals.len() - 1) {
        let delta = z_vals[i + 1] - z_vals[i];
        let alpha = 1.0 - (-coarse_sigma[i] * delta).exp();
        let weight = transmittance * alpha;
        c_expected += coarse_rgb[i] * weight;
        transmittance *= 1.0 - alpha;
    }

    let c = compute_final_color(o, d, near, far, samples_per_ray, &network);

    println!("Network computed: {:?}, expected {:?}", c, c_expected);
}
