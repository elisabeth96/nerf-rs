# LEGO NeRF weights for dependency-free Rust loaders

This directory contains float32 tensors exported from `logs/lego_example/model_200000.npy`
and `model_fine_200000.npy`, rewritten into a minimal format that can be loaded in Rust
using only the standard library.

## Layout

```
lego_rust/
  coarse/
    shapes.txt
    dense0_kernel.bin
    dense0_bias.bin
    ...
    alpha_bias.bin
  fine/
    shapes.txt
    dense0_kernel.bin
    ...
```

* Each `.bin` file stores a single tensor flattened in **row-major** order (C order) as
  little-endian `f32` values.
* `shapes.txt` lists the tensor name followed by its dimensions. One-dimensional tensors
  (biases) only have a single number after the name.

Example lines from `shapes.txt`:

```
dense0_kernel 63 256
dense0_bias 256
...
alpha_kernel 256 1
alpha_bias 1
```

The same naming scheme is used for the coarse and fine networks.

## Reading the data in Rust (no external crates)

```rust
use std::fs;
use std::path::Path;

fn load_tensor(path: &Path, dims: &[usize]) -> Vec<f32> {
    let bytes = fs::read(path).expect("read tensor");
    let mut scalars = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        scalars.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    assert_eq!(scalars.len(), dims.iter().product());
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

fn main() {
    let root = Path::new("lego_rust/coarse");
    for (name, dims) in load_shapes(&root.join("shapes.txt")) {
        let data = load_tensor(&root.join(format!("{name}.bin")), &dims);
        println!("{} has {} values", name, data.len());
    }
}
```

Feel free to reshape the returned `Vec<f32>` into whatever structure your inference
engine expects.
