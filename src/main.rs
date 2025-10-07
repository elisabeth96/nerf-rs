mod network;
mod vec3;

use network::{Layer, Matrix, Network};
use rand::Rng;
use rayon::prelude::*;
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
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

fn load_network_from_dir(dir: &Path) -> Network {
    let mut params: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for (name, dims) in load_shapes(&dir.join("shapes.txt")) {
        let data = load_tensor(&dir.join(format!("{name}.bin")), &dims);
        params.insert(name, (dims, data));
    }

    fn take_matrix(params: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> Matrix {
        let (dims, data) = params
            .remove(name)
            .unwrap_or_else(|| panic!("missing matrix parameter: {name}"));
        debug_assert!(dims.len() == 2, "matrix dims mismatch for {name}");
        debug_assert_eq!(dims[0] * dims[1], data.len());
        Matrix::new(data, dims[0] as i32, dims[1] as i32)
    }

    fn take_bias(params: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> Vec<f32> {
        let (dims, data) = params
            .remove(name)
            .unwrap_or_else(|| panic!("missing bias parameter: {name}"));
        debug_assert!(dims.len() == 1, "bias dims mismatch for {name}");
        debug_assert_eq!(dims[0], data.len());
        data
    }

    let dense_specs = [
        ("dense0_kernel", "dense0_bias"),
        ("dense1_kernel", "dense1_bias"),
        ("dense2_kernel", "dense2_bias"),
        ("dense3_kernel", "dense3_bias"),
        ("dense4_kernel", "dense4_bias"),
        ("dense5_kernel", "dense5_bias"),
        ("dense6_kernel", "dense6_bias"),
        ("dense7_kernel", "dense7_bias"),
    ];

    let layers = dense_specs
        .into_iter()
        .map(|(kernel, bias)| {
            Layer::new(
                take_matrix(&mut params, kernel),
                take_bias(&mut params, bias),
            )
        })
        .collect();

    let bottleneck = Layer::new(
        take_matrix(&mut params, "bottleneck_kernel"),
        take_bias(&mut params, "bottleneck_bias"),
    );
    let viewdirs = Layer::new(
        take_matrix(&mut params, "viewdirs_kernel"),
        take_bias(&mut params, "viewdirs_bias"),
    );
    let rgb = Layer::new(
        take_matrix(&mut params, "rgb_kernel"),
        take_bias(&mut params, "rgb_bias"),
    );
    let alpha = Layer::new(
        take_matrix(&mut params, "alpha_kernel"),
        take_bias(&mut params, "alpha_bias"),
    );

    debug_assert!(params.is_empty(), "unused parameters left after load");

    Network::new(layers, bottleneck, viewdirs, rgb, alpha)
}

fn integrate_ray(colors: &[Vec3], sigmas: &[f32], t: &[f32], far: f32) -> Vec3 {
    let n = t.len();
    if n == 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    debug_assert_eq!(colors.len(), n);
    debug_assert_eq!(sigmas.len(), n);

    let mut rgb = Vec3::new(0.0, 0.0, 0.0);
    let mut acc = 0.0f32;

    let weights = compute_weights(sigmas, t, far);
    for (color, w) in colors.iter().zip(weights.iter()) {
        rgb += *color * *w;
        acc += *w;
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

fn stratified_samples(rng: &mut impl Rng, near: f32, far: f32, count: usize) -> Vec<f32> {
    let mut t = Vec::with_capacity(count);
    if count == 0 {
        return t;
    }

    let interval = (far - near) / count as f32;
    for i in 0..count {
        let lower = near + i as f32 * interval;
        let upper = lower + interval;
        let jitter = rng.gen_range(0.0..1.0);
        t.push(lower + (upper - lower) * jitter);
    }

    t
}

fn compute_weights(sigmas: &[f32], t: &[f32], far: f32) -> Vec<f32> {
    let n = t.len();
    debug_assert_eq!(
        sigmas.len(),
        n,
        "weights require matching sigma/sample counts"
    );

    let mut weights = Vec::with_capacity(n);
    let mut transmittance = 1.0f32;

    for i in 0..n {
        let mut delta = if i + 1 < n {
            t[i + 1] - t[i]
        } else {
            far - t[i]
        };
        if delta < 0.0 {
            delta = 0.0;
        }

        let alpha = 1.0 - (-sigmas[i] * delta).exp();
        let w = transmittance * alpha;
        weights.push(w);
        transmittance *= 1.0 - alpha;

        if transmittance < 1e-4 {
            weights.extend(std::iter::repeat(0.0).take(n - i - 1));
            break;
        }
    }

    weights
}

fn midpoints(values: &[f32]) -> Vec<f32> {
    values.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect()
}

fn sample_importance(
    rng: &mut impl Rng,
    samples: &[f32],
    weights: &[f32],
    count: usize,
) -> Vec<f32> {
    if count == 0 || samples.len() < 3 || weights.len() < 3 {
        return Vec::new();
    }

    let pdf_weights = &weights[1..weights.len() - 1];
    if pdf_weights.is_empty() {
        return Vec::new();
    }

    let bins = midpoints(samples);
    if bins.len() < 2 || bins.len() != pdf_weights.len() + 1 {
        return Vec::new();
    }

    let mut adjusted: Vec<f32> = pdf_weights.iter().map(|&w| w.max(0.0) + 1e-5).collect();
    let sum: f32 = adjusted.iter().sum();
    if sum <= 0.0 {
        return Vec::new();
    }

    for w in adjusted.iter_mut() {
        *w /= sum;
    }

    let mut cdf = Vec::with_capacity(adjusted.len() + 1);
    cdf.push(0.0);
    let mut cumulative = 0.0;
    for &p in &adjusted {
        cumulative += p;
        cdf.push(cumulative);
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }

    let mut samples_out = Vec::with_capacity(count);
    for _ in 0..count {
        let u = rng.gen_range(0.0..1.0);
        let mut idx = adjusted.len() - 1;
        for j in 0..adjusted.len() {
            if u >= cdf[j] && u < cdf[j + 1] {
                idx = j;
                break;
            }
        }

        let cdf_lower = cdf[idx];
        let cdf_upper = cdf[idx + 1];
        let denom = (cdf_upper - cdf_lower).max(1e-6);
        let bin_lower = bins[idx];
        let bin_upper = bins[idx + 1];
        let t = (u - cdf_lower) / denom;
        samples_out.push(bin_lower + (bin_upper - bin_lower) * t);
    }

    samples_out
}

fn render_image(
    coarse: &Network,
    fine: &Network,
    camera: &Camera,
    fine_samples_per_ray: usize,
) -> Vec<Vec3> {
    let origin = camera.pos;
    let total_pixels = camera.nx * camera.ny;
    let coarse_samples_per_ray = camera.samples_per_ray;
    assert!(
        coarse_samples_per_ray > 0,
        "coarse samples per ray must be greater than 0"
    );

    let pixel_count = Arc::new(AtomicUsize::new(0));
    let pixel_count_clone = pixel_count.clone();

    let block_size = 8usize;
    assert_eq!(
        camera.nx % block_size,
        0,
        "image width must be multiple of block size"
    );
    assert_eq!(
        camera.ny % block_size,
        0,
        "image height must be multiple of block size"
    );

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
            for i in block_y..block_y + block_size {
                for j in block_x..block_x + block_size {
                    let dir = camera.get_ray_dir(i, j);
                    rays.push((i * camera.nx + j, dir.normalize()));
                }
            }

            let mut coarse_rng = rand::thread_rng();
            let coarse_samples: Vec<Vec<f32>> = rays
                .iter()
                .map(|_| {
                    stratified_samples(
                        &mut coarse_rng,
                        camera.near,
                        camera.far,
                        coarse_samples_per_ray,
                    )
                })
                .collect();

            let total_coarse_samples = rays.len() * coarse_samples_per_ray;
            let mut coarse_points = Matrix::zeros(3, total_coarse_samples as i32);
            let mut coarse_view_dirs = Vec::with_capacity(total_coarse_samples);

            for (ray_idx, (_, dir_hat)) in rays.iter().enumerate() {
                let samples = &coarse_samples[ray_idx];
                for (sample_idx, &ti) in samples.iter().enumerate() {
                    let col = ray_idx * coarse_samples_per_ray + sample_idx;
                    let p = origin + *dir_hat * ti;
                    coarse_points.set(0, col, p.x);
                    coarse_points.set(1, col, p.y);
                    coarse_points.set(2, col, p.z);
                    coarse_view_dirs.push(*dir_hat);
                }
            }

            let (_, coarse_sigmas) = coarse.forward_batch(&coarse_points, &coarse_view_dirs);

            let mut fine_samples_per_ray_list = Vec::with_capacity(rays.len());
            let mut fine_rng = rand::thread_rng();

            for ray_idx in 0..rays.len() {
                let start = ray_idx * coarse_samples_per_ray;
                let end = start + coarse_samples_per_ray;
                let coarse_ts = &coarse_samples[ray_idx];
                let sigmas_slice = &coarse_sigmas[start..end];
                let weights = compute_weights(sigmas_slice, coarse_ts, camera.far);

                let mut merged = coarse_ts.clone();
                let extra =
                    sample_importance(&mut fine_rng, coarse_ts, &weights, fine_samples_per_ray);
                merged.extend(extra);
                merged.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                fine_samples_per_ray_list.push(merged);
            }

            let total_fine_samples: usize = fine_samples_per_ray_list
                .iter()
                .map(|samples| samples.len())
                .sum();
            let mut fine_points = Matrix::zeros(3, total_fine_samples as i32);
            let mut fine_view_dirs = Vec::with_capacity(total_fine_samples);

            let mut fine_offsets = Vec::with_capacity(rays.len());
            let mut col = 0usize;
            for (ray_idx, samples) in fine_samples_per_ray_list.iter().enumerate() {
                fine_offsets.push(col);
                let (_, dir_hat) = rays[ray_idx];
                for &ti in samples {
                    let p = origin + dir_hat * ti;
                    fine_points.set(0, col, p.x);
                    fine_points.set(1, col, p.y);
                    fine_points.set(2, col, p.z);
                    fine_view_dirs.push(dir_hat);
                    col += 1;
                }
            }

            let (fine_colors, fine_sigmas) = fine.forward_batch(&fine_points, &fine_view_dirs);

            let mut results = Vec::with_capacity(rays.len());
            for (ray_idx, (pixel_index, _)) in rays.iter().enumerate() {
                let start = fine_offsets[ray_idx];
                let samples = &fine_samples_per_ray_list[ray_idx];
                let end = start + samples.len();
                let color = integrate_ray(
                    &fine_colors[start..end],
                    &fine_sigmas[start..end],
                    samples,
                    camera.far,
                );
                results.push((*pixel_index, color));
            }

            let processed =
                pixel_count_clone.fetch_add(rays.len(), AtomicOrdering::Relaxed) + rays.len();
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
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.join("lego_rust");
    let coarse_network = load_network_from_dir(&root.join("coarse"));
    let fine_network = load_network_from_dir(&root.join("fine"));

    let samples = load_tf_samples(&root.join("tf_reference_samples.json")).unwrap();

    let near = samples["near"].as_f64().unwrap() as f32;
    let far = samples["far"].as_f64().unwrap() as f32;
    let pos = get_vec3(&samples, "camera_origin");
    let dir = get_vec3(&samples, "camera_forward").normalize();
    let up = get_vec3(&samples, "camera_up").normalize();

    let coarse_samples_per_ray = 64;
    let fine_samples_per_ray = 128;

    println!(
        "Rendering with {} coarse samples and {} fine samples per ray",
        coarse_samples_per_ray, fine_samples_per_ray
    );

    // fill in image resolution/FOV however you like
    let cam = Camera {
        nx: 4096,
        ny: 4096,
        alpha_width: PI / 8.0,
        alpha_height: PI / 8.0,
        pos,
        dir,
        up,
        near,
        far,
        samples_per_ray: coarse_samples_per_ray,
    };

    println!("Starting image rendering...");
    let render_start = Instant::now();
    let image = render_image(&coarse_network, &fine_network, &cam, fine_samples_per_ray);
    let render_duration = render_start.elapsed();

    println!(
        "Rendering completed in {:.2} seconds",
        render_duration.as_secs_f64()
    );
    save_ppm(&Path::new("output.ppm"), cam.nx, cam.ny, &image).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(label: &str, expected: f32, actual: f32) {
        let diff = (expected - actual).abs();
        assert!(
            diff < 1e-2,
            "{} mismatch: expected {} got {} (diff {})",
            label,
            expected,
            actual,
            diff
        );
    }

    struct Example {
        ray_dir: Vec3,
        view_dir: Vec3,
        coarse_sigmas: [f32; 5],
        coarse_rgb: [[f32; 3]; 5],
        fine_sigmas: [f32; 5],
        fine_rgb: [[f32; 3]; 5],
    }

    #[test]
    fn coarse_and_fine_match_reference_examples() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let coarse = load_network_from_dir(&manifest_dir.join("lego_rust/coarse"));
        let fine = load_network_from_dir(&manifest_dir.join("lego_rust/fine"));

        let origin = Vec3::new(-0.053798322, 3.8454704, 1.2080823);
        let sample_positions = [2.0f32, 3.0, 4.0, 5.0, 6.0];

        let examples = [
            Example {
                ray_dir: Vec3 {
                    x: 0.013345719,
                    y: -0.95394367,
                    z: -0.2996883,
                },
                view_dir: Vec3 {
                    x: 0.013345721,
                    y: -0.9539438,
                    z: -0.29968834,
                },
                coarse_sigmas: [0.0, 0.0, 57.520237, 112.53807, 36.354565],
                coarse_rgb: [
                    [0.9999987, 0.9999991, 1.0],
                    [0.17455171, 0.1438829, 0.089204505],
                    [0.33823228, 0.2553625, 0.09265556],
                    [0.7298782, 0.65580726, 0.4858736],
                    [0.99999994, 0.99999994, 1.0],
                ],
                fine_sigmas: [0.0, 0.0, 116.636, 213.10716, 0.0],
                fine_rgb: [
                    [0.9989065, 0.998896, 0.999808],
                    [0.9112585, 0.94893014, 0.99545395],
                    [0.4137012, 0.33586174, 0.09175034],
                    [0.8538921, 0.77298534, 0.6327022],
                    [1.0, 1.0, 1.0],
                ],
            },
            Example {
                ray_dir: Vec3 {
                    x: -0.02204708,
                    y: -0.9975982,
                    z: -0.16230695,
                },
                view_dir: Vec3 {
                    x: -0.021808151,
                    y: -0.986787,
                    z: -0.160548,
                },
                coarse_sigmas: [0.0, 1.2744776, 0.0, 0.0, 0.0],
                coarse_rgb: [
                    [0.9975145, 0.9993601, 0.99999815],
                    [0.77613157, 0.5667896, 0.061884273],
                    [0.95516205, 0.8531487, 0.18930879],
                    [0.9998949, 0.9997925, 0.99985504],
                    [1.0, 1.0, 1.0],
                ],
                fine_sigmas: [0.0, 0.0, 0.0, 0.0, 0.0],
                fine_rgb: [
                    [0.9822932, 0.9836513, 0.9970033],
                    [0.8241439, 0.8009091, 0.56087965],
                    [0.98987126, 0.97824395, 0.88446456],
                    [0.99906117, 0.99984854, 0.9999975],
                    [1.0, 1.0, 1.0],
                ],
            },
            Example {
                ray_dir: Vec3 {
                    x: 0.22856998,
                    y: -0.8969835,
                    z: -0.471415,
                },
                view_dir: Vec3 {
                    x: 0.22003776,
                    y: -0.86350024,
                    z: -0.4538177,
                },
                coarse_sigmas: [0.0, 0.0, 2.6539133, 34.351215, 0.0],
                coarse_rgb: [
                    [1.0, 1.0, 1.0],
                    [0.9997359, 0.9998949, 0.99999803],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                fine_sigmas: [0.0, 0.0, 0.0, 0.0, 0.0],
                fine_rgb: [
                    [0.9998816, 0.9998875, 0.9999629],
                    [0.99769306, 0.99890935, 0.9998496],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
            },
        ];

        for (example_idx, example) in examples.iter().enumerate() {
            let mut points = Matrix::zeros(3, sample_positions.len() as i32);
            let mut view_dirs = Vec::with_capacity(sample_positions.len());

            for (col, &t) in sample_positions.iter().enumerate() {
                let dir = example.view_dir;
                let p = origin + example.ray_dir * t;
                points.set(0, col, p.x);
                points.set(1, col, p.y);
                points.set(2, col, p.z);
                view_dirs.push(dir);
            }

            let (coarse_colors, coarse_sigmas) = coarse.forward_batch(&points, &view_dirs);
            for (idx, &expected_sigma) in example.coarse_sigmas.iter().enumerate() {
                assert_close(
                    &format!("coarse[{example_idx}].sigma[{idx}]"),
                    expected_sigma,
                    coarse_sigmas[idx],
                );
            }
            for (idx, color) in coarse_colors.iter().enumerate() {
                let expected = example.coarse_rgb[idx];
                assert_close(
                    &format!("coarse[{example_idx}].color[{idx}].r"),
                    expected[0],
                    color.x,
                );
                assert_close(
                    &format!("coarse[{example_idx}].color[{idx}].g"),
                    expected[1],
                    color.y,
                );
                assert_close(
                    &format!("coarse[{example_idx}].color[{idx}].b"),
                    expected[2],
                    color.z,
                );
            }

            let (fine_colors, fine_sigmas) = fine.forward_batch(&points, &view_dirs);
            for (idx, &expected_sigma) in example.fine_sigmas.iter().enumerate() {
                assert_close(
                    &format!("fine[{example_idx}].sigma[{idx}]"),
                    expected_sigma,
                    fine_sigmas[idx],
                );
            }
            for (idx, color) in fine_colors.iter().enumerate() {
                let expected = example.fine_rgb[idx];
                assert_close(
                    &format!("fine[{example_idx}].color[{idx}].r"),
                    expected[0],
                    color.x,
                );
                assert_close(
                    &format!("fine[{example_idx}].color[{idx}].g"),
                    expected[1],
                    color.y,
                );
                assert_close(
                    &format!("fine[{example_idx}].color[{idx}].b"),
                    expected[2],
                    color.z,
                );
            }
        }
    }
}
