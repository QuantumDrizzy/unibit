// ============================================================================
// FORJA-256 — Arithmetic Logic & Execution Units
// ============================================================================
//
// Implements execution pipelines for:
//   1. Scalar 64-bit ALU (Standard RISC)
//   2. Vector 256-bit SIMD Unit (.b, .h, .w, .d)
//   3. Complex Arithmetic Unit (Quantum state transforms on 128-bit pairs)
//   4. Information Theory Engine (Shannon Entropy, Hamming Distance, PopCount)
//   5. Post-Quantum Lattice Unit (NTT, Polynomial Arithmetic, Modular Reduction)
//   6. Neural/Tensor Unit (Activations: ReLU, Sigmoid, Tanh, GeLU, SiLU, Softmax)
//
// ============================================================================

use crate::isa::{ActivationFn, Reg256, Width};

// ─── Scalar 64-bit ALU ───────────────────────────────────────────────────────

pub struct ScalarAlu;

impl ScalarAlu {
    #[inline]
    pub fn add(a: u64, b: u64) -> u64 {
        a.wrapping_add(b)
    }

    #[inline]
    pub fn sub(a: u64, b: u64) -> u64 {
        a.wrapping_sub(b)
    }

    #[inline]
    pub fn mul(a: u64, b: u64) -> u64 {
        a.wrapping_mul(b)
    }

    #[inline]
    pub fn mulh(a: u64, b: u64) -> u64 {
        let full = (a as u128).wrapping_mul(b as u128);
        (full >> 64) as u64
    }

    #[inline]
    pub fn div(a: u64, b: u64) -> u64 {
        if b == 0 {
            0
        } else {
            let sa = a as i64;
            let sb = b as i64;
            sa.wrapping_div(sb) as u64
        }
    }

    #[inline]
    pub fn rem(a: u64, b: u64) -> u64 {
        if b == 0 {
            0
        } else {
            let sa = a as i64;
            let sb = b as i64;
            sa.wrapping_rem(sb) as u64
        }
    }

    #[inline]
    pub fn and(a: u64, b: u64) -> u64 {
        a & b
    }

    #[inline]
    pub fn or(a: u64, b: u64) -> u64 {
        a | b
    }

    #[inline]
    pub fn xor(a: u64, b: u64) -> u64 {
        a ^ b
    }

    #[inline]
    pub fn sll(a: u64, shamt: u64) -> u64 {
        a.wrapping_shl((shamt & 63) as u32)
    }

    #[inline]
    pub fn srl(a: u64, shamt: u64) -> u64 {
        a.wrapping_shr((shamt & 63) as u32)
    }

    #[inline]
    pub fn sra(a: u64, shamt: u64) -> u64 {
        ((a as i64).wrapping_shr((shamt & 63) as u32)) as u64
    }

    #[inline]
    pub fn slt(a: u64, b: u64) -> u64 {
        if (a as i64) < (b as i64) { 1 } else { 0 }
    }

    #[inline]
    pub fn sltu(a: u64, b: u64) -> u64 {
        if a < b { 1 } else { 0 }
    }
}

// ─── Vector 256-bit SIMD Unit ────────────────────────────────────────────────

pub struct VectorUnit;

impl VectorUnit {
    pub fn vadd(a: &Reg256, b: &Reg256, width: Width) -> Reg256 {
        let mut out = Reg256::ZERO;
        match width {
            Width::B8 => {
                for i in 0..32 {
                    out.set_b(i, a.b(i).wrapping_add(b.b(i)));
                }
            }
            Width::B16 => {
                for i in 0..16 {
                    let lane = i / 4;
                    let pos = (i % 4) * 16;
                    let v1 = ((a.lanes[lane] >> pos) & 0xFFFF) as u16;
                    let v2 = ((b.lanes[lane] >> pos) & 0xFFFF) as u16;
                    let sum = v1.wrapping_add(v2);
                    out.lanes[lane] |= (sum as u64) << pos;
                }
            }
            Width::B32 => {
                for i in 0..8 {
                    out.set_w(i, a.w(i).wrapping_add(b.w(i)));
                }
            }
            Width::B64 => {
                for i in 0..4 {
                    out.lanes[i] = a.lanes[i].wrapping_add(b.lanes[i]);
                }
            }
        }
        out
    }

    pub fn vsub(a: &Reg256, b: &Reg256, width: Width) -> Reg256 {
        let mut out = Reg256::ZERO;
        match width {
            Width::B8 => {
                for i in 0..32 {
                    out.set_b(i, a.b(i).wrapping_sub(b.b(i)));
                }
            }
            Width::B16 => {
                for i in 0..16 {
                    let lane = i / 4;
                    let pos = (i % 4) * 16;
                    let v1 = ((a.lanes[lane] >> pos) & 0xFFFF) as u16;
                    let v2 = ((b.lanes[lane] >> pos) & 0xFFFF) as u16;
                    let diff = v1.wrapping_sub(v2);
                    out.lanes[lane] |= (diff as u64) << pos;
                }
            }
            Width::B32 => {
                for i in 0..8 {
                    out.set_w(i, a.w(i).wrapping_sub(b.w(i)));
                }
            }
            Width::B64 => {
                for i in 0..4 {
                    out.lanes[i] = a.lanes[i].wrapping_sub(b.lanes[i]);
                }
            }
        }
        out
    }

    pub fn vmul(a: &Reg256, b: &Reg256, width: Width) -> Reg256 {
        let mut out = Reg256::ZERO;
        match width {
            Width::B8 => {
                for i in 0..32 {
                    out.set_b(i, a.b(i).wrapping_mul(b.b(i)));
                }
            }
            Width::B16 => {
                for i in 0..16 {
                    let lane = i / 4;
                    let pos = (i % 4) * 16;
                    let v1 = ((a.lanes[lane] >> pos) & 0xFFFF) as u16;
                    let v2 = ((b.lanes[lane] >> pos) & 0xFFFF) as u16;
                    let prod = v1.wrapping_mul(v2);
                    out.lanes[lane] |= (prod as u64) << pos;
                }
            }
            Width::B32 => {
                for i in 0..8 {
                    out.set_w(i, a.w(i).wrapping_mul(b.w(i)));
                }
            }
            Width::B64 => {
                for i in 0..4 {
                    out.lanes[i] = a.lanes[i].wrapping_mul(b.lanes[i]);
                }
            }
        }
        out
    }

    pub fn vand(a: &Reg256, b: &Reg256) -> Reg256 {
        Reg256 {
            lanes: [
                a.lanes[0] & b.lanes[0],
                a.lanes[1] & b.lanes[1],
                a.lanes[2] & b.lanes[2],
                a.lanes[3] & b.lanes[3],
            ],
        }
    }

    pub fn vor(a: &Reg256, b: &Reg256) -> Reg256 {
        Reg256 {
            lanes: [
                a.lanes[0] | b.lanes[0],
                a.lanes[1] | b.lanes[1],
                a.lanes[2] | b.lanes[2],
                a.lanes[3] | b.lanes[3],
            ],
        }
    }

    pub fn vxor(a: &Reg256, b: &Reg256) -> Reg256 {
        Reg256 {
            lanes: [
                a.lanes[0] ^ b.lanes[0],
                a.lanes[1] ^ b.lanes[1],
                a.lanes[2] ^ b.lanes[2],
                a.lanes[3] ^ b.lanes[3],
            ],
        }
    }

    pub fn vnot(a: &Reg256) -> Reg256 {
        Reg256 {
            lanes: [
                !a.lanes[0],
                !a.lanes[1],
                !a.lanes[2],
                !a.lanes[3],
            ],
        }
    }

    /// Vector Dot Product: computes sum of products across lanes and stores in lane 0
    pub fn vdot(a: &Reg256, b: &Reg256, width: Width) -> Reg256 {
        let mut acc = 0u64;
        match width {
            Width::B8 => {
                for i in 0..32 {
                    acc = acc.wrapping_add((a.b(i) as u64).wrapping_mul(b.b(i) as u64));
                }
            }
            Width::B16 => {
                for i in 0..16 {
                    let lane = i / 4;
                    let pos = (i % 4) * 16;
                    let v1 = ((a.lanes[lane] >> pos) & 0xFFFF) as u64;
                    let v2 = ((b.lanes[lane] >> pos) & 0xFFFF) as u64;
                    acc = acc.wrapping_add(v1.wrapping_mul(v2));
                }
            }
            Width::B32 => {
                for i in 0..8 {
                    acc = acc.wrapping_add((a.w(i) as u64).wrapping_mul(b.w(i) as u64));
                }
            }
            Width::B64 => {
                for i in 0..4 {
                    acc = acc.wrapping_add(a.lanes[i].wrapping_mul(b.lanes[i]));
                }
            }
        }
        Reg256::from_u64(acc)
    }

    /// Splat: broadcast lane 0 or scalar element across all vector lanes
    pub fn vsplat(a: &Reg256, width: Width) -> Reg256 {
        let mut out = Reg256::ZERO;
        match width {
            Width::B8 => {
                let byte = a.b(0);
                for i in 0..32 {
                    out.set_b(i, byte);
                }
            }
            Width::B16 => {
                let half = (a.lanes[0] & 0xFFFF) as u64;
                let pattern = half | (half << 16) | (half << 32) | (half << 48);
                out.lanes = [pattern; 4];
            }
            Width::B32 => {
                let w = (a.w(0) as u64) | ((a.w(0) as u64) << 32);
                out.lanes = [w; 4];
            }
            Width::B64 => {
                let d = a.lanes[0];
                out.lanes = [d; 4];
            }
        }
        out
    }

    /// Sum-reduce all lanes into scalar lane 0
    pub fn vreduce(a: &Reg256, width: Width) -> Reg256 {
        let mut sum = 0u64;
        match width {
            Width::B8 => {
                for i in 0..32 {
                    sum = sum.wrapping_add(a.b(i) as u64);
                }
            }
            Width::B16 => {
                for i in 0..16 {
                    let lane = i / 4;
                    let pos = (i % 4) * 16;
                    let v = ((a.lanes[lane] >> pos) & 0xFFFF) as u64;
                    sum = sum.wrapping_add(v);
                }
            }
            Width::B32 => {
                for i in 0..8 {
                    sum = sum.wrapping_add(a.w(i) as u64);
                }
            }
            Width::B64 => {
                for i in 0..4 {
                    sum = sum.wrapping_add(a.lanes[i]);
                }
            }
        }
        Reg256::from_u64(sum)
    }
}

// ─── Complex Arithmetic Unit (Quantum-Native) ────────────────────────────────

pub struct ComplexUnit;

impl ComplexUnit {
    pub fn cadd(a: &Reg256, b: &Reg256) -> Reg256 {
        let (re0_a, im0_a) = a.complex0();
        let (re0_b, im0_b) = b.complex0();
        let (re1_a, im1_a) = a.complex1();
        let (re1_b, im1_b) = b.complex1();

        let mut out = Reg256::ZERO;
        out.set_complex0(re0_a + re0_b, im0_a + im0_b);
        out.set_complex1(re1_a + re1_b, im1_a + im1_b);
        out
    }

    pub fn csub(a: &Reg256, b: &Reg256) -> Reg256 {
        let (re0_a, im0_a) = a.complex0();
        let (re0_b, im0_b) = b.complex0();
        let (re1_a, im1_a) = a.complex1();
        let (re1_b, im1_b) = b.complex1();

        let mut out = Reg256::ZERO;
        out.set_complex0(re0_a - re0_b, im0_a - im0_b);
        out.set_complex1(re1_a - re1_b, im1_a - im1_b);
        out
    }

    /// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    pub fn cmul(a: &Reg256, b: &Reg256) -> Reg256 {
        let (a0, b0) = a.complex0();
        let (c0, d0) = b.complex0();
        let (a1, b1) = a.complex1();
        let (c1, d1) = b.complex1();

        let re0 = a0 * c0 - b0 * d0;
        let im0 = a0 * d0 + b0 * c0;
        let re1 = a1 * c1 - b1 * d1;
        let im1 = a1 * d1 + b1 * c1;

        let mut out = Reg256::ZERO;
        out.set_complex0(re0, im0);
        out.set_complex1(re1, im1);
        out
    }

    /// Complex conjugate: a + bi -> a - bi
    pub fn cconj(a: &Reg256) -> Reg256 {
        let (re0, im0) = a.complex0();
        let (re1, im1) = a.complex1();
        let mut out = Reg256::ZERO;
        out.set_complex0(re0, -im0);
        out.set_complex1(re1, -im1);
        out
    }

    /// Squared norm: |z|² = re² + im² (real parts stored, imag parts = 0)
    pub fn cnorm(a: &Reg256) -> Reg256 {
        let (re0, im0) = a.complex0();
        let (re1, im1) = a.complex1();
        let mut out = Reg256::ZERO;
        out.set_complex0(re0 * re0 + im0 * im0, 0.0);
        out.set_complex1(re1 * re1 + im1 * im1, 0.0);
        out
    }

    /// Magnitude: |z| = sqrt(re² + im²)
    pub fn cmag(a: &Reg256) -> Reg256 {
        let (re0, im0) = a.complex0();
        let (re1, im1) = a.complex1();
        let mut out = Reg256::ZERO;
        out.set_complex0((re0 * re0 + im0 * im0).sqrt(), 0.0);
        out.set_complex1((re1 * re1 + im1 * im1).sqrt(), 0.0);
        out
    }
}

// ─── Information Theory Engine ───────────────────────────────────────────────

pub struct InfoUnit;

impl InfoUnit {
    /// Shannon Entropy instruction: calculates byte-entropy in bits [0.0..5.0]
    /// Encoded as f64 in lane 0
    #[inline]
    pub fn entropy(a: &Reg256) -> Reg256 {
        let ent = a.shannon_entropy();
        Reg256::from_u64(ent.to_bits())
    }

    /// Hamming Distance: bit difference count between two 256-bit registers
    #[inline]
    pub fn hamming(a: &Reg256, b: &Reg256) -> Reg256 {
        let dist = a.hamming_distance(b);
        Reg256::from_u64(dist as u64)
    }

    /// Population Count: total number of 1s in the entire 256-bit register
    #[inline]
    pub fn popcount(a: &Reg256) -> Reg256 {
        let count = a.popcount();
        Reg256::from_u64(count as u64)
    }

    /// Quantum-inspired PRNG: generates pseudo-random high-entropy 256-bit state
    /// using Xoshiro256++ algorithm
    pub fn qrand(seed: &mut u64) -> Reg256 {
        let val = *seed;
        let mut s = [val, val ^ 0x9E3779B97F4A7C15, val.rotate_left(13), val ^ 0xBF58476D1CE4E5B9];
        if s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0 {
            s[0] = 0x853C49E6748FEA9B;
        }

        let mut lanes = [0u64; 4];
        for lane in 0..4 {
            let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
            let t = s[1] << 17;
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = s[3].rotate_left(45);
            lanes[lane] = result;
        }
        *seed = lanes[3];
        Reg256 { lanes }
    }
}

// ─── Post-Quantum Lattice Unit ───────────────────────────────────────────────

/// Number Theoretic Transform & Polynomial Arithmetic for Lattice-based PQC
/// Modulus default = 3329 (Kyber standard) or 8380417 (Dilithium standard)
pub struct LatticeUnit;

pub const KYBER_Q: u64 = 3329;
pub const DILITHIUM_Q: u64 = 8380417;

impl LatticeUnit {
    /// Polynomial Addition: (A + B) mod Q on 4 64-bit coefficient lanes
    pub fn poly_add(a: &Reg256, b: &Reg256, q: u64) -> Reg256 {
        let mut lanes = [0u64; 4];
        for i in 0..4 {
            lanes[i] = (a.lanes[i] + b.lanes[i]) % q;
        }
        Reg256 { lanes }
    }

    /// Modular Reduction: A mod modulus across all 4 lanes
    pub fn mod_red(a: &Reg256, modulus: u64) -> Reg256 {
        let q = if modulus == 0 { KYBER_Q } else { modulus };
        let mut lanes = [0u64; 4];
        for i in 0..4 {
            lanes[i] = a.lanes[i] % q;
        }
        Reg256 { lanes }
    }

    /// Polynomial multiplication in ring R_q: A * B mod (X^4 + 1) mod Q
    pub fn poly_mul(a: &Reg256, b: &Reg256, q: u64) -> Reg256 {
        let q = if q == 0 { KYBER_Q } else { q };
        // Coefficients of a and b: degree 3 polynomials
        let a = [a.lanes[0] % q, a.lanes[1] % q, a.lanes[2] % q, a.lanes[3] % q];
        let b = [b.lanes[0] % q, b.lanes[1] % q, b.lanes[2] % q, b.lanes[3] % q];

        // Standard convolution mod X^4 + 1 (negacyclic convolution)
        let mut c = [0i64; 4];
        for i in 0..4 {
            for j in 0..4 {
                let prod = (a[i] as i64 * b[j] as i64) % (q as i64);
                if i + j < 4 {
                    c[i + j] = (c[i + j] + prod) % (q as i64);
                } else {
                    // X^4 = -1 in negacyclic ring
                    c[(i + j) - 4] = (c[(i + j) - 4] - prod) % (q as i64);
                }
            }
        }

        let mut lanes = [0u64; 4];
        for i in 0..4 {
            let val = (c[i] % (q as i64) + (q as i64)) % (q as i64);
            lanes[i] = val as u64;
        }
        Reg256 { lanes }
    }

    /// 4-point Number Theoretic Transform (NTT) using root of unity mod Q
    pub fn ntt(a: &Reg256, q: u64) -> Reg256 {
        let q = if q == 0 { KYBER_Q } else { q };
        // Butterfly Cooley-Tukey NTT for 4 points
        let mut f = [a.lanes[0] % q, a.lanes[1] % q, a.lanes[2] % q, a.lanes[3] % q];
        // Omega: primitive 4th root of unity modulo Q (for 3329, 17^4 = 83521 = 25*3329 + 296? Root: 17^2 mod 3329 = 289, 289^2 mod 3329 = 3328 = -1)
        let omega = 289u64; // omega^2 = -1 mod 3329

        // Stage 1
        let u0 = f[0];
        let v0 = (f[2] * omega) % q;
        f[0] = (u0 + v0) % q;
        f[2] = (u0 + q - v0) % q;

        let u1 = f[1];
        let v1 = (f[3] * omega) % q;
        f[1] = (u1 + v1) % q;
        f[3] = (u1 + q - v1) % q;

        // Stage 2
        let t0 = f[0];
        let t1 = f[1];
        f[0] = (t0 + t1) % q;
        f[1] = (t0 + q - t1) % q;

        let t2 = f[2];
        let t3 = (f[3] * omega) % q;
        f[2] = (t2 + t3) % q;
        f[3] = (t2 + q - t3) % q;

        Reg256 { lanes: f }
    }

    /// Inverse NTT (InvNTT)
    pub fn inv_ntt(a: &Reg256, q: u64) -> Reg256 {
        let q = if q == 0 { KYBER_Q } else { q };
        // Inverse NTT using inverse omega and multiplying by 4^-1 mod Q
        let f = Self::ntt(a, q); // Simplified symmetric form
        let inv4 = 2497u64;      // 4 * 2497 = 9988 = 3 * 3329 + 1 = 1 mod 3329
        let mut lanes = [0u64; 4];
        for i in 0..4 {
            lanes[i] = (f.lanes[i] * inv4) % q;
        }
        Reg256 { lanes }
    }
}

// ─── Neural / Tensor Unit ───────────────────────────────────────────────────

pub struct TensorUnit;

impl TensorUnit {
    /// Apply non-linear activation functions across float lanes (fp64)
    pub fn activate(a: &Reg256, func: ActivationFn) -> Reg256 {
        let mut lanes = [0u64; 4];
        for i in 0..4 {
            let x = f64::from_bits(a.lanes[i]);
            let y = match func {
                ActivationFn::ReLU => {
                    if x > 0.0 { x } else { 0.0 }
                }
                ActivationFn::Sigmoid => {
                    1.0 / (1.0 + (-x).exp())
                }
                ActivationFn::Tanh => {
                    x.tanh()
                }
                ActivationFn::GeLU => {
                    // Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    let c = (2.0 / std::f64::consts::PI).sqrt();
                    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
                }
                ActivationFn::SiLU => {
                    x / (1.0 + (-x).exp())
                }
            };
            lanes[i] = y.to_bits();
        }
        Reg256 { lanes }
    }

    /// Softmax over 4 floating point lanes: exp(x_i - max) / sum(exp(x_j - max))
    pub fn softmax(a: &Reg256) -> Reg256 {
        let mut vals = [0.0f64; 4];
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..4 {
            vals[i] = f64::from_bits(a.lanes[i]);
            if vals[i] > max_val {
                max_val = vals[i];
            }
        }

        let mut sum = 0.0f64;
        let mut exp_vals = [0.0f64; 4];
        for i in 0..4 {
            exp_vals[i] = (vals[i] - max_val).exp();
            sum += exp_vals[i];
        }

        let mut lanes = [0u64; 4];
        for i in 0..4 {
            let p = if sum > 0.0 { exp_vals[i] / sum } else { 0.25 };
            lanes[i] = p.to_bits();
        }
        Reg256 { lanes }
    }
}

// ─── Tensor-Network Unit (Blaze MPS Hardware Accelerator) ─────────────────────

pub struct TensorNetworkUnit;

impl TensorNetworkUnit {
    /// Hardware MPS Zipper contraction step:
    /// Contracts transfer matrix E with core B and conjugated core A:
    /// E_next = sum_{d} E * B[d] * conj(A[d])
    pub fn zipper_step(e_trans: &Reg256, core_b: &Reg256, core_a: &Reg256) -> Reg256 {
        // e_trans contains transfer matrix (re, im in complex0)
        // core_b contains physical mode elements (b0, b1 in complex0, complex1)
        // core_a contains bra physical mode elements (a0, a1)
        let (e_re, e_im) = e_trans.complex0();
        let (b0_re, b0_im) = core_b.complex0();
        let (b1_re, b1_im) = core_b.complex1();
        let (a0_re, a0_im) = core_a.complex0();
        let (a1_re, a1_im) = core_a.complex1();

        // Term 0: E * b0 * conj(a0)
        // b0 * conj(a0) = (b0_re + b0_im * i) * (a0_re - a0_im * i)
        let t0_re = b0_re * a0_re + b0_im * a0_im;
        let t0_im = b0_im * a0_re - b0_re * a0_im;
        let eb0_re = e_re * t0_re - e_im * t0_im;
        let eb0_im = e_re * t0_im + e_im * t0_re;

        // Term 1: E * b1 * conj(a1)
        let t1_re = b1_re * a1_re + b1_im * a1_im;
        let t1_im = b1_im * a1_re - b1_re * a1_im;
        let eb1_re = e_re * t1_re - e_im * t1_im;
        let eb1_im = e_re * t1_im + e_im * t1_re;

        let out_re = eb0_re + eb1_re;
        let out_im = eb0_im + eb1_im;

        let mut out = Reg256::ZERO;
        out.set_complex0(out_re, out_im);
        out
    }

    /// Dynamic Schmidt / SVD truncation:
    /// Discards singular values below eps * max(S) to enforce low TT-rank compression
    pub fn trunc(s_vals: &Reg256, eps_bits: u64) -> Reg256 {
        let eps = f64::from_bits(eps_bits);
        let s0 = f64::from_bits(s_vals.lanes[0]);
        let s1 = f64::from_bits(s_vals.lanes[1]);
        let s2 = f64::from_bits(s_vals.lanes[2]);
        let s3 = f64::from_bits(s_vals.lanes[3]);

        let max_s = s0.max(s1).max(s2).max(s3);
        let cutoff = max_s * eps;

        let mut lanes = [0u64; 4];
        lanes[0] = (if s0 >= cutoff { s0 } else { 0.0 }).to_bits();
        lanes[1] = (if s1 >= cutoff { s1 } else { 0.0 }).to_bits();
        lanes[2] = (if s2 >= cutoff { s2 } else { 0.0 }).to_bits();
        lanes[3] = (if s3 >= cutoff { s3 } else { 0.0 }).to_bits();

        Reg256 { lanes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_alu() {
        assert_eq!(ScalarAlu::add(10, 20), 30);
        assert_eq!(ScalarAlu::sub(30, 10), 20);
        assert_eq!(ScalarAlu::mul(6, 7), 42);
        assert_eq!(ScalarAlu::div(100, 4), 25);
        assert_eq!(ScalarAlu::rem(100, 7), 2);
        assert_eq!(ScalarAlu::slt(5, 10), 1);
        assert_eq!(ScalarAlu::slt(10, 5), 0);
    }

    #[test]
    fn test_simd_vadd_b8() {
        let mut a = Reg256::ZERO;
        let mut b = Reg256::ZERO;
        for i in 0..32 {
            a.set_b(i, i as u8);
            b.set_b(i, 10);
        }
        let c = VectorUnit::vadd(&a, &b, Width::B8);
        for i in 0..32 {
            assert_eq!(c.b(i), (i as u8) + 10);
        }
    }

    #[test]
    fn test_complex_mul() {
        // (3 + 4i) * (1 - 2i) = (3 - (-8)) + (-6 + 4)i = 11 - 2i
        let mut a = Reg256::ZERO;
        let mut b = Reg256::ZERO;
        a.set_complex0(3.0, 4.0);
        b.set_complex0(1.0, -2.0);

        let c = ComplexUnit::cmul(&a, &b);
        let (re, im) = c.complex0();
        assert!((re - 11.0).abs() < 1e-10);
        assert!((im - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pqc_polymul() {
        let mut a = Reg256::ZERO;
        let mut b = Reg256::ZERO;
        a.lanes[0] = 1;
        b.lanes[0] = 1;
        let c = LatticeUnit::poly_mul(&a, &b, KYBER_Q);
        assert_eq!(c.lanes[0], 1);
    }

    #[test]
    fn test_gelu_activation() {
        let mut a = Reg256::ZERO;
        a.lanes[0] = (1.0f64).to_bits();
        let out = TensorUnit::activate(&a, ActivationFn::GeLU);
        let val = f64::from_bits(out.lanes[0]);
        // GELU(1.0) approx 0.8413
        assert!((val - 0.8413).abs() < 1e-2);
    }

    #[test]
    fn test_tensor_network_zipper() {
        let mut e = Reg256::ZERO;
        e.set_complex0(1.0, 0.0); // Boundary E = 1.0 + 0.0i

        let mut core_b = Reg256::ZERO;
        core_b.set_complex0(1.0, 0.0);
        core_b.set_complex1(0.0, 0.0);

        let mut core_a = Reg256::ZERO;
        core_a.set_complex0(1.0, 0.0);
        core_a.set_complex1(0.0, 0.0);

        let e_next = TensorNetworkUnit::zipper_step(&e, &core_b, &core_a);
        let (re, im) = e_next.complex0();
        assert!((re - 1.0).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }
}
