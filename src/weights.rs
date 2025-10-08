#[cfg(target_arch = "wasm32")]
pub mod embedded {
    pub fn shapes(network: &str) -> &'static str {
        match network {
            "coarse" => include_str!("../lego_rust/coarse/shapes.txt"),
            "fine" => include_str!("../lego_rust/fine/shapes.txt"),
            _ => panic!("unknown network {network}"),
        }
    }

    pub fn tensor_bytes(network: &str, name: &str) -> &'static [u8] {
        match (network, name) {
            ("coarse", "alpha_bias.bin") => include_bytes!("../lego_rust/coarse/alpha_bias.bin"),
            ("coarse", "alpha_kernel.bin") => {
                include_bytes!("../lego_rust/coarse/alpha_kernel.bin")
            }
            ("coarse", "bottleneck_bias.bin") => {
                include_bytes!("../lego_rust/coarse/bottleneck_bias.bin")
            }
            ("coarse", "bottleneck_kernel.bin") => {
                include_bytes!("../lego_rust/coarse/bottleneck_kernel.bin")
            }
            ("coarse", "dense0_bias.bin") => include_bytes!("../lego_rust/coarse/dense0_bias.bin"),
            ("coarse", "dense0_kernel.bin") => {
                include_bytes!("../lego_rust/coarse/dense0_kernel.bin")
            }
            ("coarse", "dense1_bias.bin") => include_bytes!("../lego_rust/coarse/dense1_bias.bin"),
            ("coarse", "dense1_kernel.bin") => {
                include_bytes!("../lego_rust/coarse/dense1_kernel.bin")
            }
            ("coarse", "dense2_bias.bin") => include_bytes!("../lego_rust/coarse/dense2_bias.bin"),
            ("coarse", "dense2_kernel.bin") => {
                include_bytes!("../lego_rust/coarse/dense2_kernel.bin")
            }
            ("coarse", "dense3_bias.bin") => include_bytes!("../lego_rust/coarse/dense3_bias.bin"),
            ("coarse", "dense3_kernel.bin") => {
                include_bytes!("../lego_rust/coarse/dense3_kernel.bin")
            }
            ("coarse", "dense4_bias.bin") => include_bytes!("../lego_rust/coarse/dense4_bias.bin"),
            ("coarse", "dense4_kernel.bin") => {
                include_bytes!("../lego_rust/coarse/dense4_kernel.bin")
            }
            ("coarse", "dense5_bias.bin") => include_bytes!("../lego_rust/coarse/dense5_bias.bin"),
            ("coarse", "dense5_kernel.bin") => {
                include_bytes!("../lego_rust/coarse/dense5_kernel.bin")
            }
            ("coarse", "dense6_bias.bin") => include_bytes!("../lego_rust/coarse/dense6_bias.bin"),
            ("coarse", "dense6_kernel.bin") => {
                include_bytes!("../lego_rust/coarse/dense6_kernel.bin")
            }
            ("coarse", "dense7_bias.bin") => include_bytes!("../lego_rust/coarse/dense7_bias.bin"),
            ("coarse", "dense7_kernel.bin") => {
                include_bytes!("../lego_rust/coarse/dense7_kernel.bin")
            }
            ("coarse", "rgb_bias.bin") => include_bytes!("../lego_rust/coarse/rgb_bias.bin"),
            ("coarse", "rgb_kernel.bin") => include_bytes!("../lego_rust/coarse/rgb_kernel.bin"),
            ("coarse", "viewdirs_bias.bin") => {
                include_bytes!("../lego_rust/coarse/viewdirs_bias.bin")
            }
            ("coarse", "viewdirs_kernel.bin") => {
                include_bytes!("../lego_rust/coarse/viewdirs_kernel.bin")
            }
            ("fine", "alpha_bias.bin") => include_bytes!("../lego_rust/fine/alpha_bias.bin"),
            ("fine", "alpha_kernel.bin") => include_bytes!("../lego_rust/fine/alpha_kernel.bin"),
            ("fine", "bottleneck_bias.bin") => {
                include_bytes!("../lego_rust/fine/bottleneck_bias.bin")
            }
            ("fine", "bottleneck_kernel.bin") => {
                include_bytes!("../lego_rust/fine/bottleneck_kernel.bin")
            }
            ("fine", "dense0_bias.bin") => include_bytes!("../lego_rust/fine/dense0_bias.bin"),
            ("fine", "dense0_kernel.bin") => include_bytes!("../lego_rust/fine/dense0_kernel.bin"),
            ("fine", "dense1_bias.bin") => include_bytes!("../lego_rust/fine/dense1_bias.bin"),
            ("fine", "dense1_kernel.bin") => include_bytes!("../lego_rust/fine/dense1_kernel.bin"),
            ("fine", "dense2_bias.bin") => include_bytes!("../lego_rust/fine/dense2_bias.bin"),
            ("fine", "dense2_kernel.bin") => include_bytes!("../lego_rust/fine/dense2_kernel.bin"),
            ("fine", "dense3_bias.bin") => include_bytes!("../lego_rust/fine/dense3_bias.bin"),
            ("fine", "dense3_kernel.bin") => include_bytes!("../lego_rust/fine/dense3_kernel.bin"),
            ("fine", "dense4_bias.bin") => include_bytes!("../lego_rust/fine/dense4_bias.bin"),
            ("fine", "dense4_kernel.bin") => include_bytes!("../lego_rust/fine/dense4_kernel.bin"),
            ("fine", "dense5_bias.bin") => include_bytes!("../lego_rust/fine/dense5_bias.bin"),
            ("fine", "dense5_kernel.bin") => include_bytes!("../lego_rust/fine/dense5_kernel.bin"),
            ("fine", "dense6_bias.bin") => include_bytes!("../lego_rust/fine/dense6_bias.bin"),
            ("fine", "dense6_kernel.bin") => include_bytes!("../lego_rust/fine/dense6_kernel.bin"),
            ("fine", "dense7_bias.bin") => include_bytes!("../lego_rust/fine/dense7_bias.bin"),
            ("fine", "dense7_kernel.bin") => include_bytes!("../lego_rust/fine/dense7_kernel.bin"),
            ("fine", "rgb_bias.bin") => include_bytes!("../lego_rust/fine/rgb_bias.bin"),
            ("fine", "rgb_kernel.bin") => include_bytes!("../lego_rust/fine/rgb_kernel.bin"),
            ("fine", "viewdirs_bias.bin") => include_bytes!("../lego_rust/fine/viewdirs_bias.bin"),
            ("fine", "viewdirs_kernel.bin") => {
                include_bytes!("../lego_rust/fine/viewdirs_kernel.bin")
            }
            _ => panic!("unknown tensor {network}/{name}"),
        }
    }

    pub fn tf_reference_samples() -> &'static str {
        include_str!("../lego_rust/tf_reference_samples.json")
    }
}
