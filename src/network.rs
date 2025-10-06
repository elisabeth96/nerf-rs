use crate::vec3::Vec3;

#[derive(Clone)]
pub struct Matrix {
    elements: Vec<f32>,
    rows: i32,
    cols: i32,
}

impl Matrix {
    pub fn new(elements: Vec<f32>, rows: i32, cols: i32) -> Self {
        debug_assert_eq!(elements.len(), (rows * cols) as usize);
        Self {
            elements,
            rows,
            cols,
        }
    }

    pub fn empty() -> Self {
        Self {
            elements: Vec::new(),
            rows: 0,
            cols: 0,
        }
    }

    pub fn zeros(rows: i32, cols: i32) -> Self {
        Self {
            elements: vec![0.0; (rows * cols) as usize],
            rows,
            cols,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    pub fn cols(&self) -> usize {
        self.cols as usize
    }

    pub fn data(&self) -> &[f32] {
        &self.elements
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.elements
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        let cols = self.cols();
        self.elements[row * cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        let cols = self.cols();
        self.elements[row * cols + col] = value;
    }

    pub fn row_slice(&self, row: usize) -> &[f32] {
        let cols = self.cols();
        let start = row * cols;
        let end = start + cols;
        &self.elements[start..end]
    }
}

pub struct Layer {
    weight: Matrix,
    bias: Vec<f32>,
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
    fn forward_matrix(&self, h: &Matrix, act: ActivationType) -> Matrix {
        self.forward_accelerate(h, act)
    }

    #[cfg(not(target_os = "macos"))]
    fn forward_matrix(&self, h: &Matrix, act: ActivationType) -> Matrix {
        self.forward_fallback(h, act)
    }

    #[cfg(target_os = "macos")]
    fn forward_accelerate(&self, h: &Matrix, act: ActivationType) -> Matrix {
        use cblas_sys::{CBLAS_ORDER, CBLAS_TRANSPOSE, cblas_sgemm};

        let input_dim = self.weight.rows as i32;
        let output_dim = self.weight.cols as i32;
        debug_assert_eq!(h.rows as i32, input_dim);

        let batch = h.cols as i32;
        let mut result = Matrix::zeros(output_dim, batch);
        Self::fill_with_bias(&mut result, &self.bias);

        unsafe {
            cblas_sgemm(
                CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                output_dim,
                batch,
                input_dim,
                1.0,
                self.weight.data().as_ptr(),
                self.weight.cols as i32,
                h.data().as_ptr(),
                h.cols as i32,
                1.0,
                result.data_mut().as_mut_ptr(),
                result.cols as i32,
            );
        }

        Self::apply_activation(result.data_mut(), act);
        result
    }

    #[cfg_attr(target_os = "macos", allow(dead_code))]
    fn forward_fallback(&self, h: &Matrix, act: ActivationType) -> Matrix {
        let rows = self.weight.rows();
        let cols = self.weight.cols();
        let batch = h.cols();
        debug_assert_eq!(h.rows(), rows);

        let mut result = Matrix::zeros(cols as i32, batch as i32);
        Self::fill_with_bias(&mut result, &self.bias);

        for in_idx in 0..rows {
            let weight_row = &self.weight.data()[in_idx * cols..(in_idx + 1) * cols];
            for col_idx in 0..batch {
                let h_val = h.get(in_idx, col_idx);
                for out_idx in 0..cols {
                    let current = result.get(out_idx, col_idx) + weight_row[out_idx] * h_val;
                    result.set(out_idx, col_idx, current);
                }
            }
        }

        Self::apply_activation(result.data_mut(), act);
        result
    }

    fn fill_with_bias(matrix: &mut Matrix, bias: &[f32]) {
        debug_assert_eq!(matrix.rows(), bias.len());
        let cols = matrix.cols();
        for row in 0..bias.len() {
            let row_start = row * cols;
            let row_slice = &mut matrix.data_mut()[row_start..row_start + cols];
            for value in row_slice.iter_mut() {
                *value = bias[row];
            }
        }
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
    alpha: Layer,
}

impl Network {
    pub fn new(
        layers: Vec<Layer>,
        bottleneck: Layer,
        viewdirs: Layer,
        rgb: Layer,
        alpha: Layer,
    ) -> Self {
        Self {
            layers,
            bottleneck,
            viewdirs,
            rgb,
            alpha,
        }
    }

    pub fn forward_batch(&self, points: &Matrix, view_dirs: &[Vec3]) -> (Vec<Vec3>, Vec<f32>) {
        let batch = points.cols();
        if batch == 0 {
            return (Vec::new(), Vec::new());
        }
        debug_assert_eq!(batch, view_dirs.len());

        let h_0 = positional_encoding_batch(points, 10);
        let mut h = h_0.clone();
        for i in 0..5 {
            h = self.layers[i].forward_matrix(&h, ActivationType::ReLU);
        }
        let h4 = h.clone();
        h = concat_rows(&h_0, &h4);
        for i in 5..8 {
            h = self.layers[i].forward_matrix(&h, ActivationType::ReLU);
        }
        let h8 = h.clone();

        let sigma_matrix = self.alpha.forward_matrix(&h8, ActivationType::ReLU);

        let bottleneck = self.bottleneck.forward_matrix(&h8, ActivationType::None);
        let viewdir_matrix = positional_encoding_dirs(view_dirs, 4);
        let q = concat_rows(&bottleneck, &viewdir_matrix);

        let c_hidden = self.viewdirs.forward_matrix(&q, ActivationType::ReLU);
        let c_matrix = self.rgb.forward_matrix(&c_hidden, ActivationType::Sigmoid);

        let mut colors = Vec::with_capacity(batch);
        for col in 0..batch {
            colors.push(Vec3::new(
                c_matrix.get(0, col),
                c_matrix.get(1, col),
                c_matrix.get(2, col),
            ));
        }

        let sigma_values = sigma_matrix.row_slice(0).to_vec();

        (colors, sigma_values)
    }
}

fn concat_rows(a: &Matrix, b: &Matrix) -> Matrix {
    debug_assert_eq!(a.cols(), b.cols());
    let rows = (a.rows() + b.rows()) as i32;
    let cols = a.cols() as i32;
    let mut result = Matrix::zeros(rows, cols);

    let cols_usize = a.cols();
    for row in 0..a.rows() {
        let src = a.row_slice(row);
        let dst_start = row * cols_usize;
        result.data_mut()[dst_start..dst_start + cols_usize].copy_from_slice(src);
    }

    for row in 0..b.rows() {
        let src = b.row_slice(row);
        let dst_row = row + a.rows();
        let dst_start = dst_row * cols_usize;
        result.data_mut()[dst_start..dst_start + cols_usize].copy_from_slice(src);
    }

    result
}

fn positional_encoding_batch(points: &Matrix, n: i32) -> Matrix {
    debug_assert_eq!(points.rows(), 3);
    let cols = points.cols();
    let encoded_dim = 3 + 6 * n as usize;
    let mut encoded = Matrix::zeros(encoded_dim as i32, cols as i32);

    for col in 0..cols {
        encoded.set(0, col, points.get(0, col));
        encoded.set(1, col, points.get(1, col));
        encoded.set(2, col, points.get(2, col));

        let mut f = 1.0f32;
        let mut row = 3;
        for _ in 0..n {
            for axis in 0..3 {
                let value = points.get(axis, col);
                encoded.set(row, col, (f * value).sin());
                row += 1;
            }
            for axis in 0..3 {
                let value = points.get(axis, col);
                encoded.set(row, col, (f * value).cos());
                row += 1;
            }
            f *= 2.0;
        }
    }

    encoded
}

fn positional_encoding_dirs(view_dirs: &[Vec3], n: i32) -> Matrix {
    let cols = view_dirs.len();
    let encoded_dim = 3 + 6 * n as usize;
    let mut encoded = Matrix::zeros(encoded_dim as i32, cols as i32);

    for (col, dir) in view_dirs.iter().enumerate() {
        encoded.set(0, col, dir.x);
        encoded.set(1, col, dir.y);
        encoded.set(2, col, dir.z);

        let mut f = 1.0f32;
        let mut row = 3;
        for _ in 0..n {
            for axis in 0..3 {
                let value = match axis {
                    0 => dir.x,
                    1 => dir.y,
                    _ => dir.z,
                };
                encoded.set(row, col, (f * value).sin());
                row += 1;
            }
            for axis in 0..3 {
                let value = match axis {
                    0 => dir.x,
                    1 => dir.y,
                    _ => dir.z,
                };
                encoded.set(row, col, (f * value).cos());
                row += 1;
            }
            f *= 2.0;
        }
    }

    encoded
}
