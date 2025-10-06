use crate::vec3::Vec3;

pub struct Matrix {
    elements: Vec<f32>,
    rows: i32,
    cols: i32
}

impl Matrix {
    pub fn new(elements: Vec<f32>, rows: i32, cols: i32) -> Self {
        Self { elements, rows, cols }
    }

    pub fn empty() -> Self {
        Self { elements: Vec::new(), rows: 0, cols: 0 }
    }
}

pub struct Layer {
    weight: Matrix,
    bias: Vec<f32>
}

#[derive(Copy, Clone)]
enum ActivationType {
    ReLU, 
    Sigmoid,
    None,
}

impl Layer {
    pub fn new(weight: Matrix, bias: Vec<f32>) -> Self {
        Self { weight, bias }
    }

    #[cfg(target_os = "macos")]
    fn forward(&self, h: &Vec<f32>, act: ActivationType) -> Vec<f32> {
        self.forward_accelerate(h, act)
    }

    #[cfg(not(target_os = "macos"))]
    fn forward(&self, h: &Vec<f32>, act: ActivationType) -> Vec<f32> {
        self.forward_fallback(h, act)
    }

    #[cfg(target_os = "macos")]
    fn forward_accelerate(&self, h: &Vec<f32>, act: ActivationType) -> Vec<f32> {
        use cblas_sys::{cblas_sgemv, CBLAS_ORDER, CBLAS_TRANSPOSE};

        let input_dim = self.weight.rows as usize;
        let output_dim = self.weight.cols as usize;
        debug_assert_eq!(h.len(), input_dim);

        let mut result = self.bias.clone();
        debug_assert_eq!(result.len(), output_dim);

        unsafe {
            cblas_sgemv(
                CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasTrans,
                self.weight.rows,
                self.weight.cols,
                1.0,
                self.weight.elements.as_ptr(),
                self.weight.cols,
                h.as_ptr(),
                1,
                1.0,
                result.as_mut_ptr(),
                1,
            );
        }

        Self::apply_activation(&mut result, act);
        result
    }

    #[cfg_attr(target_os = "macos", allow(dead_code))]
    fn forward_fallback(&self, h: &Vec<f32>, act: ActivationType) -> Vec<f32> {
        let rows = self.weight.rows as usize;
        let cols = self.weight.cols as usize;
        let mut result = self.bias.clone();

        for in_idx in 0..rows {
            let h_val = h[in_idx];
            let row_offset = in_idx * cols;
            for out_idx in 0..cols {
                result[out_idx] += self.weight.elements[row_offset + out_idx] * h_val;
            }
        }

        Self::apply_activation(&mut result, act);
        result
    }

    fn apply_activation(values: &mut [f32], act: ActivationType) {
        for v in values.iter_mut() {
            *v = match act {
                ActivationType::ReLU => (*v).max(0.0),
                ActivationType::Sigmoid => 1.0 / (1.0 + (-*v).exp()),
                ActivationType::None => *v,
            };
        }
    }
}

pub struct Network {
    layers: Vec<Layer>,
    bottleneck: Layer,
    viewdirs: Layer,
    rgb: Layer,
    alpha: Layer
}

impl Network {
    pub fn new(layers: Vec<Layer>, bottleneck: Layer, viewdirs: Layer, rgb: Layer, alpha: Layer) -> Self {
        Self { layers, bottleneck, viewdirs, rgb, alpha }
    }

    pub fn forward(&self, p: &Vec3, view_dir: &Vec3) -> (Vec3, f32) {
        let h_0 = positional_encoding(p, 10);
        let y_dir = positional_encoding(view_dir, 4);
        let mut h = h_0.clone();
        for i in 0..5 {
            h = self.layers[i].forward(&h, ActivationType::ReLU);
        }
        let h4 = h.clone();
        let s = concat(&h_0, &h4);
        h = s;
        for i in 5..8 {
            h = self.layers[i].forward(&h, ActivationType::ReLU)
        }
        let h8 = h.clone();
        let sigma = self.alpha.forward(&h8, ActivationType::ReLU);
        let bottleneck = self.bottleneck.forward(&h8, ActivationType::None);
        let q = concat(&bottleneck, &y_dir);
        let c_hidden = self.viewdirs.forward(&q, ActivationType::ReLU);
        let c = self.rgb.forward(&c_hidden, ActivationType::Sigmoid);
        let c_final = Vec3::new(c[0], c[1], c[2]);
        (c_final, sigma[0])
    }
     
}

fn concat(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
    let mut result = a.clone();
    result.extend(b);
    result
}

fn positional_encoding(p: &Vec3, n: i32) -> Vec<f32> {
    let base = [p.x, p.y, p.z];
    let mut v = Vec::with_capacity(3 + 6 * n as usize);
    v.extend(base);
    let mut f = 1.0f32;
    for _ in 0..n {
        v.extend(base.map(|c| (f * c).sin()));
        v.extend(base.map(|c| (f * c).cos()));
        f *= 2.0;
    }
    v
}
