// ============================================================================
// Unibit — Arithmetic Logic & Execution Units
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

// Lane loops are indexed deliberately. In the SIMD, NTT and softmax kernels
// the loop variable is the mathematical subscript (lane i, coefficient j,
// frequency k in omega^(j*k)), and several loops read two arrays at unequal
// offsets. Rewriting them as iterator chains would obscure the formulas these
// functions are checked against.
#![allow(clippy::needless_range_loop)]

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
                    let v1 = (a.lanes[lane] >> pos) & 0xFFFF;
                    let v2 = (b.lanes[lane] >> pos) & 0xFFFF;
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
                let half = a.lanes[0] & 0xFFFF;
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
                    let v = (a.lanes[lane] >> pos) & 0xFFFF;
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

/// Ring degree of R_q = Z_q[X]/(X^N + 1): one coefficient per 64-bit lane.
pub const RING_N: usize = 4;

// ─── Modular arithmetic helpers ─────────────────────────────────────────────
//
// All products go through u128 so the unit stays correct for any prime
// modulus below 2^63, not just the small Kyber one.

/// (a * b) mod q
#[inline]
fn mod_mul(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

/// base^exp mod q by square-and-multiply
fn mod_pow(base: u64, mut exp: u64, q: u64) -> u64 {
    let mut acc = 1 % q;
    let mut b = base % q;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mod_mul(acc, b, q);
        }
        b = mod_mul(b, b, q);
        exp >>= 1;
    }
    acc
}

/// a^-1 mod q via Fermat's little theorem. Requires q prime, which holds for
/// every lattice-PQC modulus.
#[inline]
fn mod_inv(a: u64, q: u64) -> u64 {
    mod_pow(a, q - 2, q)
}

impl LatticeUnit {
    /// Derives psi, a primitive 2N-th root of unity mod q (i.e. psi^N == -1).
    ///
    /// psi is what makes the transform *negacyclic*: it folds the ring's
    /// wrap-around rule X^N = -1 into the transform, so that the NTT is
    /// consistent with `poly_mul` rather than with plain cyclic convolution.
    ///
    /// Instead of hardcoding per-modulus constants (which silently produce
    /// garbage the moment q changes), psi is derived. For any base x, the
    /// element x^((q-1)/2N) has order dividing 2N, and has order exactly 2N
    /// with probability phi(2N)/2N = 1/2. Ascending trial from x = 2 therefore
    /// terminates after ~2 iterations and is deterministic for a given q.
    ///
    /// Returns `None` when no such root exists (2N does not divide q-1), so
    /// the caller traps instead of computing nonsense.
    pub fn negacyclic_root(q: u64) -> Option<u64> {
        let two_n = 2 * RING_N as u64;
        if q < 3 || !(q - 1).is_multiple_of(two_n) {
            return None;
        }
        let exp = (q - 1) / two_n;
        let neg_one = q - 1;
        for x in 2..q {
            let psi = mod_pow(x, exp, q);
            if mod_pow(psi, RING_N as u64, q) == neg_one {
                return Some(psi);
            }
        }
        None
    }

    /// Polynomial Addition: (A + B) mod Q on N 64-bit coefficient lanes
    pub fn poly_add(a: &Reg256, b: &Reg256, q: u64) -> Reg256 {
        let q = if q == 0 { KYBER_Q } else { q };
        let mut lanes = [0u64; RING_N];
        for i in 0..RING_N {
            // Reduce each operand *before* adding: raw lanes are arbitrary
            // 64-bit values, so a + b would overflow near u64::MAX.
            lanes[i] = (a.lanes[i] % q + b.lanes[i] % q) % q;
        }
        Reg256 { lanes }
    }

    /// Modular Reduction: A mod modulus across all N lanes
    pub fn mod_red(a: &Reg256, modulus: u64) -> Reg256 {
        let q = if modulus == 0 { KYBER_Q } else { modulus };
        let mut lanes = [0u64; RING_N];
        for i in 0..RING_N {
            lanes[i] = a.lanes[i] % q;
        }
        Reg256 { lanes }
    }

    /// Polynomial multiplication in the ring R_q = Z_q[X]/(X^N + 1):
    /// schoolbook negacyclic convolution, the reference `ntt` is checked against.
    pub fn poly_mul(a: &Reg256, b: &Reg256, q: u64) -> Reg256 {
        let q = if q == 0 { KYBER_Q } else { q };
        let av: Vec<u64> = (0..RING_N).map(|i| a.lanes[i] % q).collect();
        let bv: Vec<u64> = (0..RING_N).map(|i| b.lanes[i] % q).collect();

        let mut c = [0u64; RING_N];
        for i in 0..RING_N {
            for j in 0..RING_N {
                let prod = mod_mul(av[i], bv[j], q);
                let idx = (i + j) % RING_N;
                if i + j < RING_N {
                    c[idx] = (c[idx] + prod) % q;
                } else {
                    // X^N = -1: terms that wrap around change sign
                    c[idx] = (c[idx] + q - prod) % q;
                }
            }
        }
        Reg256 { lanes: c }
    }

    /// Negacyclic Number Theoretic Transform over R_q = Z_q[X]/(X^N + 1):
    ///
    /// ```text
    ///     A[k] = sum_{j=0}^{N-1} a[j] * psi^j * omega^(j*k)   (mod q)
    /// ```
    ///
    /// with omega = psi^2 a primitive N-th root of unity. The psi^j pre-twist
    /// is what yields the convolution theorem
    ///
    /// ```text
    ///     NTT(a * b mod X^N+1) == NTT(a) .* NTT(b)
    /// ```
    ///
    /// which `test_ntt_convolution_theorem` verifies against `poly_mul`.
    ///
    /// Evaluated as a direct O(N^2) sum: at N = 4 a radix-2 Cooley-Tukey
    /// decomposition saves 4 multiplies and costs far more in clarity.
    ///
    /// Returns `None` if q admits no 2N-th root of unity (see `negacyclic_root`).
    pub fn ntt(a: &Reg256, q: u64) -> Option<Reg256> {
        let q = if q == 0 { KYBER_Q } else { q };
        let psi = Self::negacyclic_root(q)?;
        let omega = mod_mul(psi, psi, q);

        // Pre-twist: a'[j] = a[j] * psi^j
        let mut twisted = [0u64; RING_N];
        let mut psi_j = 1 % q;
        for j in 0..RING_N {
            twisted[j] = mod_mul(a.lanes[j] % q, psi_j, q);
            psi_j = mod_mul(psi_j, psi, q);
        }

        let mut lanes = [0u64; RING_N];
        for k in 0..RING_N {
            let mut acc = 0u64;
            for j in 0..RING_N {
                let tw = mod_pow(omega, (j * k) as u64, q);
                acc = (acc + mod_mul(twisted[j], tw, q)) % q;
            }
            lanes[k] = acc;
        }
        Some(Reg256 { lanes })
    }

    /// Inverse negacyclic NTT: an exact left inverse of [`LatticeUnit::ntt`].
    ///
    /// ```text
    ///     a[j] = N^-1 * psi^-j * sum_{k=0}^{N-1} A[k] * omega^(-j*k)   (mod q)
    /// ```
    ///
    /// Modular inverses use Fermat's little theorem, valid because every
    /// lattice-PQC modulus (3329, 8380417, 12289, ...) is prime.
    pub fn inv_ntt(a: &Reg256, q: u64) -> Option<Reg256> {
        let q = if q == 0 { KYBER_Q } else { q };
        let psi = Self::negacyclic_root(q)?;
        let omega = mod_mul(psi, psi, q);
        let inv_omega = mod_inv(omega, q);
        let inv_psi = mod_inv(psi, q);
        let inv_n = mod_inv(RING_N as u64, q);

        let mut lanes = [0u64; RING_N];
        let mut inv_psi_j = 1 % q;
        for j in 0..RING_N {
            let mut acc = 0u64;
            for k in 0..RING_N {
                let tw = mod_pow(inv_omega, (j * k) as u64, q);
                acc = (acc + mod_mul(a.lanes[k] % q, tw, q)) % q;
            }
            // Undo the pre-twist and scale by N^-1
            lanes[j] = mod_mul(mod_mul(acc, inv_psi_j, q), inv_n, q);
            inv_psi_j = mod_mul(inv_psi_j, inv_psi, q);
        }
        Some(Reg256 { lanes })
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

    /// 2x2 f64 matrix multiply, row-major: lanes = [m00, m01, m10, m11].
    ///
    /// A 256-bit register holds exactly four f64, so 2x2 is the largest dense
    /// matrix that fits in one operand. Anything larger needs multiple
    /// registers and does not belong in a single-register instruction.
    pub fn matmul2x2(a: &Reg256, b: &Reg256) -> Reg256 {
        let m = |r: &Reg256, i: usize| f64::from_bits(r.lanes[i]);
        let (a00, a01, a10, a11) = (m(a, 0), m(a, 1), m(a, 2), m(a, 3));
        let (b00, b01, b10, b11) = (m(b, 0), m(b, 1), m(b, 2), m(b, 3));

        Reg256 {
            lanes: [
                (a00 * b00 + a01 * b10).to_bits(),
                (a00 * b01 + a01 * b11).to_bits(),
                (a10 * b00 + a11 * b10).to_bits(),
                (a10 * b01 + a11 * b11).to_bits(),
            ],
        }
    }

    /// f64 dot product over the four lanes, result in lane 0.
    ///
    /// Distinct from `VectorUnit::vdot`, which is integer and wraps. This is
    /// the float path the activation/attention units actually need.
    pub fn dot_f64(a: &Reg256, b: &Reg256) -> Reg256 {
        let mut acc = 0.0f64;
        for i in 0..4 {
            acc += f64::from_bits(a.lanes[i]) * f64::from_bits(b.lanes[i]);
        }
        Reg256::from_u64(acc.to_bits())
    }
}

// ─── Tensor-Network Unit (MPS contraction) ───────────────────────────────────

pub struct TensorNetworkUnit;

impl TensorNetworkUnit {
    /// One MPS zipper contraction step at **bond dimension 1**:
    ///
    /// ```text
    ///     E_next = E * sum_{d} B[d] * conj(A[d])
    /// ```
    ///
    /// [KNOWN_LIMIT] chi = 1 only. E is a scalar, not a transfer matrix, and
    /// the cores carry one complex amplitude per physical index d in {0, 1}.
    /// A chi = 2 step needs a 2x2 complex transfer matrix (8 f64 = 512 bits),
    /// which does not fit in one 256-bit operand; generalising it requires
    /// multi-register operands the ISA does not have. What this does compute,
    /// exactly, is the overlap <A|B> of two product states, accumulated one
    /// site at a time — see `test_zipper_matches_product_state_overlap`.
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

    /// One MPS zipper contraction step at **bond dimension 2**.
    ///
    /// The chi = 1 step above wastes most of its operands: a complex scalar is
    /// 128 bits of a 256-bit register. At chi = 2 nothing fits at f64 -- the
    /// transfer matrix is 512 bits and a core is 1024 -- so this step is mixed
    /// precision, and the split is not arbitrary:
    ///
    /// - **E is f32.** It is the accumulator, carried across every site, so its
    ///   error compounds. A 2x2 complex f32 is exactly 256 bits: one register,
    ///   nothing wasted.
    /// - **Cores are int8 codes with per-bond scales.** They are static data,
    ///   quantized once. 16 codes (128 bits) + 2 f32 scales (64 bits) = 192 bits.
    ///
    /// That split is measured, not assumed:
    /// `test_zipper2_error_budget_is_dominated_by_the_cores` runs the ablation
    /// over 200 random chi = 2 MPS of 8 sites and reports
    ///
    /// ```text
    ///     int8 cores      fidelity   mean 0.999962, worst 0.999921
    ///     int8 cores      norm drift mean 2.5e-3,   worst 1.4e-2
    ///     f32 accumulator            worst relative error 5.0e-7
    /// ```
    ///
    /// So the cores are the entire error budget and the accumulator is free,
    /// four orders of magnitude cheaper -- the opposite of the naive guess.
    /// The direction of the state survives (fidelity 0.9999) while its
    /// magnitude drifts by a fraction of a percent.
    ///
    /// Random MPS are used deliberately: they are near-maximally entangled at
    /// chi = 2, so they are the hard case: a structured ground state would
    /// quantise more forgivingly than the numbers above.
    ///
    /// Layout of `e_trans` and of the result, row-major over [a][b]:
    ///
    /// ```text
    ///     word 2*(2a+b)      = Re E[a][b]   (f32)
    ///     word 2*(2a+b) + 1  = Im E[a][b]
    /// ```
    ///
    /// Layout of a core, over indices (l, d, r) each in {0, 1}:
    ///
    /// ```text
    ///     byte 2*((2l+d)*2+r)      = Re code  (i8)
    ///     byte 2*((2l+d)*2+r) + 1  = Im code
    ///     word 4 + r               = scale of right bond r  (f32)
    /// ```
    ///
    /// Ragged boundary cores -- (1,2,2) at the head, (2,2,1) at the tail -- are
    /// zero-padded into the uniform block and E starts at `[[1,0],[0,0]]`. That
    /// padding reproduces the overlap exactly; it was validated against a
    /// reference contraction before this function existed, because a wrong
    /// padding does not fail loudly, it returns a plausible wrong number.
    ///
    /// Every product term of the contraction is covered: `tools/mutation_test.py`
    /// perturbs each of the eight in turn and all eight are caught by
    /// `cargo test`.
    ///
    /// [KNOWN_LIMIT] chi = 4 does not fit at any precision. Its transfer matrix
    /// needs 32 codes plus a scale (288 bits) and its cores need 544, both past
    /// 256. A 256-bit word tops out at chi = 2. That is the ceiling of the
    /// architecture, not a gap in this implementation: raising it means 512-bit
    /// operands, which is a different ISA.
    pub fn zipper2_step(e_trans: &Reg256, core_b: &Reg256, core_a: &Reg256) -> Reg256 {
        let b = Self::unpack_core(core_b);
        let a = Self::unpack_core(core_a);

        // t[aL][d][bR] = sum_bL E[aL][bL] * B[bL][d][bR]
        let mut t = [[[(0.0f32, 0.0f32); 2]; 2]; 2];
        for al in 0..2 {
            for d in 0..2 {
                for br in 0..2 {
                    let (mut re, mut im) = (0.0f32, 0.0f32);
                    for bl in 0..2 {
                        let (e_re, e_im) = Self::e_at(e_trans, al, bl);
                        let (b_re, b_im) = b[bl][d][br];
                        re += e_re * b_re - e_im * b_im;
                        im += e_re * b_im + e_im * b_re;
                    }
                    t[al][d][br] = (re, im);
                }
            }
        }

        // E'[aR][bR] = sum_{aL,d} conj(A[aL][d][aR]) * t[aL][d][bR]
        let mut out = Reg256::ZERO;
        for ar in 0..2 {
            for br in 0..2 {
                let (mut re, mut im) = (0.0f32, 0.0f32);
                for al in 0..2 {
                    for d in 0..2 {
                        let (a_re, a_im) = a[al][d][ar];
                        let (t_re, t_im) = t[al][d][br];
                        re += a_re * t_re + a_im * t_im;
                        im += a_re * t_im - a_im * t_re;
                    }
                }
                let w = 2 * (2 * ar + br);
                out.set_f32_at(w, re);
                out.set_f32_at(w + 1, im);
            }
        }
        out
    }

    /// Read one entry of a chi = 2 transfer matrix.
    #[inline]
    pub fn e_at(e: &Reg256, a: usize, b: usize) -> (f32, f32) {
        let w = 2 * (2 * (a & 1) + (b & 1));
        (e.f32_at(w), e.f32_at(w + 1))
    }

    /// The boundary transfer matrix `[[1,0],[0,0]]` a zipper chain starts from.
    pub fn e_boundary() -> Reg256 {
        let mut e = Reg256::ZERO;
        e.set_f32_at(0, 1.0);
        e
    }

    /// Pack a chi = 2 core, indexed `[l][d][r]`, into one register.
    ///
    /// Symmetric affine quantisation at per-bond granularity:
    /// `code = round(x / scale)` clipped to +-127, with one scale per right
    /// bond. Per-bond costs one extra f32 over a single per-core scale and
    /// still fits (192 bits vs 160), and it measured better on longer chains,
    /// so it is what the instruction uses.
    pub fn pack_core(core: &[[[(f64, f64); 2]; 2]; 2]) -> Reg256 {
        const LIM: f64 = 127.0;
        let mut out = Reg256::ZERO;
        for r in 0..2 {
            let mut peak = 0.0f64;
            for l in 0..2 {
                for d in 0..2 {
                    let (re, im) = core[l][d][r];
                    peak = peak.max(re.abs()).max(im.abs());
                }
            }
            // An all-zero bond has no scale to speak of; 1.0 keeps it at zero
            // instead of producing a NaN.
            let scale = if peak > 0.0 { peak / LIM } else { 1.0 };
            out.set_f32_at(4 + r, scale as f32);
            for l in 0..2 {
                for d in 0..2 {
                    let (re, im) = core[l][d][r];
                    let idx = 2 * ((2 * l + d) * 2 + r);
                    out.set_i8_at(idx, (re / scale).round().clamp(-LIM, LIM) as i8);
                    out.set_i8_at(idx + 1, (im / scale).round().clamp(-LIM, LIM) as i8);
                }
            }
        }
        out
    }

    /// Decode a packed core back to complex f32, indexed `[l][d][r]`.
    fn unpack_core(c: &Reg256) -> [[[(f32, f32); 2]; 2]; 2] {
        let scale = [c.f32_at(4), c.f32_at(5)];
        let mut out = [[[(0.0f32, 0.0f32); 2]; 2]; 2];
        for l in 0..2 {
            for d in 0..2 {
                for r in 0..2 {
                    let idx = 2 * ((2 * l + d) * 2 + r);
                    out[l][d][r] = (c.i8_at(idx) as f32 * scale[r],
                                    c.i8_at(idx + 1) as f32 * scale[r]);
                }
            }
        }
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

    /// Deterministic xorshift, so failures are reproducible.
    fn rng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn random_poly(state: &mut u64, q: u64) -> Reg256 {
        let mut r = Reg256::ZERO;
        for i in 0..RING_N {
            r.lanes[i] = rng(state) % q;
        }
        r
    }

    #[test]
    fn test_pqc_polymul_identity() {
        // 1 * b == b in R_q, for every coefficient.
        let mut one = Reg256::ZERO;
        one.lanes[0] = 1;
        let mut st = 0x2545F491_4F6CDD1D;
        for _ in 0..64 {
            let b = random_poly(&mut st, KYBER_Q);
            let c = LatticeUnit::poly_mul(&one, &b, KYBER_Q);
            assert_eq!(c.lanes, b.lanes);
        }
    }

    #[test]
    fn test_pqc_polymul_negacyclic_wrap() {
        // X^3 * X == X^4 == -1 (mod X^4 + 1), i.e. q-1 in lane 0.
        let mut x3 = Reg256::ZERO;
        x3.lanes[3] = 1;
        let mut x1 = Reg256::ZERO;
        x1.lanes[1] = 1;
        let c = LatticeUnit::poly_mul(&x3, &x1, KYBER_Q);
        assert_eq!(c.lanes, [KYBER_Q - 1, 0, 0, 0]);
    }

    #[test]
    fn test_negacyclic_root_properties() {
        for &q in &[KYBER_Q, DILITHIUM_Q, 12289] {
            let psi = LatticeUnit::negacyclic_root(q)
                .unwrap_or_else(|| panic!("no 2N-th root of unity mod {}", q));
            // psi^N == -1 (primitive 2N-th root), hence psi^2N == 1
            assert_eq!(mod_pow(psi, RING_N as u64, q), q - 1, "q={}", q);
            assert_eq!(mod_pow(psi, 2 * RING_N as u64, q), 1, "q={}", q);
        }
        // 2N must divide q-1; 7 has no 8th root of unity.
        assert_eq!(LatticeUnit::negacyclic_root(7), None);
    }

    #[test]
    fn test_ntt_roundtrip() {
        for &q in &[KYBER_Q, DILITHIUM_Q] {
            let mut st = 0x9E3779B9_7F4A7C15;
            for _ in 0..256 {
                let a = random_poly(&mut st, q);
                let spectrum = LatticeUnit::ntt(&a, q).expect("forward ntt");
                let back = LatticeUnit::inv_ntt(&spectrum, q).expect("inverse ntt");
                assert_eq!(back.lanes, a.lanes, "invntt(ntt(a)) != a for q={}", q);
            }
        }
    }

    #[test]
    fn test_ntt_convolution_theorem() {
        // The property that makes an NTT worth having:
        // NTT(a * b mod X^N+1) == NTT(a) .* NTT(b), checked against poly_mul.
        for &q in &[KYBER_Q, DILITHIUM_Q] {
            let mut st = 0xDEADBEEF_CAFEF00D;
            for _ in 0..256 {
                let a = random_poly(&mut st, q);
                let b = random_poly(&mut st, q);

                let lhs = LatticeUnit::ntt(&LatticeUnit::poly_mul(&a, &b, q), q).unwrap();

                let na = LatticeUnit::ntt(&a, q).unwrap();
                let nb = LatticeUnit::ntt(&b, q).unwrap();
                let mut rhs = Reg256::ZERO;
                for i in 0..RING_N {
                    rhs.lanes[i] = mod_mul(na.lanes[i], nb.lanes[i], q);
                }

                assert_eq!(lhs.lanes, rhs.lanes, "convolution theorem broken for q={}", q);
            }
        }
    }

    #[test]
    fn test_ntt_rejects_unsupported_modulus() {
        // 2N does not divide 7-1, so the unit must refuse rather than guess.
        assert!(LatticeUnit::ntt(&Reg256::ZERO, 7).is_none());
        assert!(LatticeUnit::inv_ntt(&Reg256::ZERO, 7).is_none());
    }

    #[test]
    fn test_poly_add_no_overflow() {
        // Unreduced lanes near u64::MAX must not overflow the addition.
        let mut a = Reg256::ZERO;
        let mut b = Reg256::ZERO;
        for i in 0..RING_N {
            a.lanes[i] = u64::MAX;
            b.lanes[i] = u64::MAX;
        }
        let c = LatticeUnit::poly_add(&a, &b, KYBER_Q);
        let expected = (u64::MAX % KYBER_Q + u64::MAX % KYBER_Q) % KYBER_Q;
        for i in 0..RING_N {
            assert_eq!(c.lanes[i], expected);
        }
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
    fn test_matmul2x2_against_reference() {
        // [1 2; 3 4] * [5 6; 7 8] = [19 22; 43 50]
        let a = Reg256 { lanes: [1.0f64.to_bits(), 2.0f64.to_bits(), 3.0f64.to_bits(), 4.0f64.to_bits()] };
        let b = Reg256 { lanes: [5.0f64.to_bits(), 6.0f64.to_bits(), 7.0f64.to_bits(), 8.0f64.to_bits()] };
        let c = TensorUnit::matmul2x2(&a, &b);
        let got: Vec<f64> = (0..4).map(|i| f64::from_bits(c.lanes[i])).collect();
        assert_eq!(got, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul2x2_identity_is_neutral() {
        let id = Reg256 { lanes: [1.0f64.to_bits(), 0.0f64.to_bits(), 0.0f64.to_bits(), 1.0f64.to_bits()] };
        let mut st = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..64 {
            let mut m = Reg256::ZERO;
            for i in 0..4 {
                m.lanes[i] = ((rng(&mut st) % 2000) as f64 / 16.0 - 62.5).to_bits();
            }
            assert_eq!(TensorUnit::matmul2x2(&m, &id).lanes, m.lanes);
            assert_eq!(TensorUnit::matmul2x2(&id, &m).lanes, m.lanes);
        }
    }

    #[test]
    fn test_dot_f64() {
        let a = Reg256 { lanes: [1.0f64.to_bits(), 2.0f64.to_bits(), 3.0f64.to_bits(), 4.0f64.to_bits()] };
        let b = Reg256 { lanes: [0.5f64.to_bits(), (-1.0f64).to_bits(), 2.0f64.to_bits(), 0.25f64.to_bits()] };
        // 0.5 - 2.0 + 6.0 + 1.0 = 5.5
        let got = f64::from_bits(TensorUnit::dot_f64(&a, &b).as_u64());
        assert!((got - 5.5).abs() < 1e-12, "got {}", got);
    }

    #[test]
    fn test_zipper_matches_product_state_overlap() {
        // Chained zipper steps must reproduce <A|B> for two product states,
        // computed independently as prod_i sum_d conj(a_i[d]) * b_i[d].
        let sites: [([f64; 4], [f64; 4]); 3] = [
            ([0.6, 0.1, 0.8, -0.2], [0.3, 0.4, -0.5, 0.7]),
            ([-0.2, 0.9, 0.4, 0.3], [0.8, -0.1, 0.2, 0.6]),
            ([0.5, -0.5, 0.1, 0.9], [-0.7, 0.2, 0.3, 0.4]),
        ];

        // Independent reference, plain complex arithmetic on f64 pairs.
        let (mut ref_re, mut ref_im) = (1.0f64, 0.0f64);
        for (a, b) in sites.iter() {
            let (mut t_re, mut t_im) = (0.0f64, 0.0f64);
            for d in 0..2 {
                let (ar, ai) = (a[2 * d], a[2 * d + 1]);
                let (br, bi) = (b[2 * d], b[2 * d + 1]);
                // b * conj(a)
                t_re += br * ar + bi * ai;
                t_im += bi * ar - br * ai;
            }
            let (nr, ni) = (ref_re * t_re - ref_im * t_im, ref_re * t_im + ref_im * t_re);
            ref_re = nr;
            ref_im = ni;
        }

        // Same contraction driven through the instruction's execution unit.
        let mut e = Reg256::ZERO;
        e.set_complex0(1.0, 0.0);
        for (a, b) in sites.iter() {
            let mut core_a = Reg256::ZERO;
            core_a.set_complex0(a[0], a[1]);
            core_a.set_complex1(a[2], a[3]);
            let mut core_b = Reg256::ZERO;
            core_b.set_complex0(b[0], b[1]);
            core_b.set_complex1(b[2], b[3]);
            e = TensorNetworkUnit::zipper_step(&e, &core_b, &core_a);
        }

        let (got_re, got_im) = e.complex0();
        assert!((got_re - ref_re).abs() < 1e-12, "re: {} vs {}", got_re, ref_re);
        assert!((got_im - ref_im).abs() < 1e-12, "im: {} vs {}", got_im, ref_im);
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


    // ─── chi = 2 zipper ──────────────────────────────────────────────────────

    // ── Mixed-precision error budget of the chi = 2 zipper ──────────────────
    //
    // `zipper2_step` splits its 256 bits between an f32 accumulator and int8
    // cores. These helpers measure what that split actually costs, so the
    // numbers quoted in the docs are reproducible here rather than borrowed.

    type Core2 = [[[(f64, f64); 2]; 2]; 2];

    /// A random chi = 2 core, normalised so a long chain neither blows up nor
    /// underflows. Uniform entries make a near-maximally entangled MPS, which
    /// is the hard case for quantisation, not a structured ground state.
    fn random_core(state: &mut u64) -> Core2 {
        let mut c = [[[(0.0f64, 0.0f64); 2]; 2]; 2];
        let mut norm = 0.0f64;
        for l in 0..2 {
            for d in 0..2 {
                for r in 0..2 {
                    let unit = |st: &mut u64| (rng(st) as f64 / u64::MAX as f64) * 2.0 - 1.0;
                    let re = unit(state);
                    let im = unit(state);
                    c[l][d][r] = (re, im);
                    norm += re * re + im * im;
                }
            }
        }
        let k = 1.0 / norm.sqrt();
        for l in 0..2 {
            for d in 0..2 {
                for r in 0..2 {
                    c[l][d][r].0 *= k;
                    c[l][d][r].1 *= k;
                }
            }
        }
        c
    }

    /// Exact f64 reference contraction, index-for-index the same as
    /// `zipper2_step`: E'[aR][bR] = sum conj(A[aL][d][aR]) E[aL][bL] B[bL][d][bR].
    /// No f32, no quantisation — this is the ground truth the instruction is
    /// measured against.
    fn overlap_f64(a: &[Core2], b: &[Core2]) -> (f64, f64) {
        let mut e = [[(0.0f64, 0.0f64); 2]; 2];
        e[0][0] = (1.0, 0.0);
        for (ca, cb) in a.iter().zip(b.iter()) {
            let mut t = [[[(0.0f64, 0.0f64); 2]; 2]; 2];
            for al in 0..2 {
                for d in 0..2 {
                    for br in 0..2 {
                        let (mut re, mut im) = (0.0, 0.0);
                        for bl in 0..2 {
                            let (e_re, e_im) = e[al][bl];
                            let (b_re, b_im) = cb[bl][d][br];
                            re += e_re * b_re - e_im * b_im;
                            im += e_re * b_im + e_im * b_re;
                        }
                        t[al][d][br] = (re, im);
                    }
                }
            }
            let mut next = [[(0.0f64, 0.0f64); 2]; 2];
            for ar in 0..2 {
                for br in 0..2 {
                    let (mut re, mut im) = (0.0, 0.0);
                    for al in 0..2 {
                        for d in 0..2 {
                            let (a_re, a_im) = ca[al][d][ar];
                            let (t_re, t_im) = t[al][d][br];
                            re += a_re * t_re + a_im * t_im;
                            im += a_re * t_im - a_im * t_re;
                        }
                    }
                    next[ar][br] = (re, im);
                }
            }
            e = next;
        }
        e[0][0]
    }

    /// Round-trip a core through the instruction's int8 quantiser, so the
    /// effect of quantisation can be measured in exact f64 arithmetic.
    fn quantise_roundtrip(core: &Core2) -> Core2 {
        let packed = TensorNetworkUnit::pack_core(core);
        let scale = [packed.f32_at(4) as f64, packed.f32_at(5) as f64];
        let mut out = [[[(0.0f64, 0.0f64); 2]; 2]; 2];
        for l in 0..2 {
            for d in 0..2 {
                for r in 0..2 {
                    let idx = 2 * ((2 * l + d) * 2 + r);
                    out[l][d][r] = (
                        packed.i8_at(idx) as f64 * scale[r],
                        packed.i8_at(idx + 1) as f64 * scale[r],
                    );
                }
            }
        }
        out
    }

    fn modulus(z: (f64, f64)) -> f64 {
        (z.0 * z.0 + z.1 * z.1).sqrt()
    }

    #[test]
    fn test_zipper2_error_budget_is_dominated_by_the_cores() {
        // Ablation over random chi = 2 MPS. Two error sources, measured apart:
        //   cores: int8 quantisation, evaluated in exact f64
        //   accumulator: the f32 carry, evaluated on already-quantised cores
        // The claim under test is that the cores are the whole budget and the
        // f32 accumulator is free.
        const SITES: usize = 8;
        const TRIALS: usize = 200;

        let mut st = 0x5DEECE66D_u64 | 1;
        let (mut worst_fidelity, mut worst_acc_err) = (1.0f64, 0.0f64);
        let (mut worst_norm_drift, mut drift_sum) = (0.0f64, 0.0f64);
        let mut fidelity_sum = 0.0f64;

        for _ in 0..TRIALS {
            let exact: Vec<Core2> = (0..SITES).map(|_| random_core(&mut st)).collect();
            let quantised: Vec<Core2> = exact.iter().map(quantise_roundtrip).collect();

            // State fidelity under quantisation:
            //     F = |<A|A_q>| / sqrt(<A|A> <A_q|A_q>)
            let aa = modulus(overlap_f64(&exact, &exact));
            let qq = modulus(overlap_f64(&quantised, &quantised));
            let aq = modulus(overlap_f64(&exact, &quantised));
            assert!(aa > 0.0 && qq > 0.0, "degenerate random MPS");
            let fidelity = aq / (aa * qq).sqrt();

            fidelity_sum += fidelity;
            worst_fidelity = worst_fidelity.min(fidelity);

            // Norm drift, tracked apart from fidelity: quantisation can preserve
            // the direction of the state while still rescaling its magnitude.
            let drift = ((qq / aa).sqrt() - 1.0).abs();
            worst_norm_drift = worst_norm_drift.max(drift);
            drift_sum += drift;

            // The f32 accumulator, isolated: identical (quantised) cores on both
            // paths, so the only difference left is the carry precision.
            let reference = overlap_f64(&quantised, &quantised);
            let mut e = TensorNetworkUnit::e_boundary();
            let packed: Vec<Reg256> = exact.iter().map(TensorNetworkUnit::pack_core).collect();
            for core in packed.iter() {
                e = TensorNetworkUnit::zipper2_step(&e, core, core);
            }
            let (got_re, got_im) = TensorNetworkUnit::e_at(&e, 0, 0);
            let got = (got_re as f64, got_im as f64);
            let denom = modulus(reference).max(f64::MIN_POSITIVE);
            let acc_err = modulus((got.0 - reference.0, got.1 - reference.1)) / denom;
            worst_acc_err = worst_acc_err.max(acc_err);
        }

        let mean_fidelity = fidelity_sum / TRIALS as f64;
        let mean_drift = drift_sum / TRIALS as f64;
        println!(
            "chi=2 zipper over {} random MPS of {} sites:",
            TRIALS, SITES
        );
        println!(
            "  int8 cores      -> fidelity  mean {:.6}, worst {:.6}",
            mean_fidelity, worst_fidelity
        );
        println!(
            "  int8 cores      -> norm drift mean {:.3e}, worst {:.3e}",
            mean_drift, worst_norm_drift
        );
        println!(
            "  f32 accumulator -> worst relative error {:.3e}",
            worst_acc_err
        );

        // The cores carry the loss, and it stays in the third decimal or better.
        assert!(
            worst_fidelity > 0.99,
            "int8 cores lost more than expected: worst fidelity {:.6}",
            worst_fidelity
        );
        // Norm drift stays sub-percent on average and never runs away.
        assert!(
            worst_norm_drift < 0.05,
            "int8 norm drift {:.3e} exceeds the measured envelope",
            worst_norm_drift
        );
        // The f32 carry must be orders of magnitude cheaper than the cores.
        assert!(
            worst_acc_err < 1e-3,
            "f32 accumulator error {:.3e} is not negligible",
            worst_acc_err
        );
        assert!(
            worst_acc_err < 1.0 - worst_fidelity,
            "the accumulator ({:.3e}) is no longer dominated by the cores ({:.3e})",
            worst_acc_err,
            1.0 - worst_fidelity
        );
    }

    /// A core with only bond 0 populated is a chi = 1 core wearing a chi = 2
    /// costume, so the two instructions must agree on it. Entries are exact
    /// multiples of the int8 step (k/127), so quantization is lossless here and
    /// the only difference left is f32 against f64 -- which is the point: it
    /// isolates the accumulator's cost from the cores'.
    #[test]
    fn test_zipper2_reduces_to_zipper1_on_a_single_bond() {
        let b0 = (1.0, 64.0 / 127.0);
        let b1 = (-32.0 / 127.0, 96.0 / 127.0);
        let a0 = (96.0 / 127.0, -1.0);
        let a1 = (48.0 / 127.0, 16.0 / 127.0);

        // chi = 1 form
        let mut e1 = Reg256::ZERO;
        e1.set_complex0(1.0, 0.0);
        let mut cb = Reg256::ZERO;
        cb.set_complex0(b0.0, b0.1);
        cb.set_complex1(b1.0, b1.1);
        let mut ca = Reg256::ZERO;
        ca.set_complex0(a0.0, a0.1);
        ca.set_complex1(a1.0, a1.1);
        let (want_re, want_im) = TensorNetworkUnit::zipper_step(&e1, &cb, &ca).complex0();

        // chi = 2 form: everything on bond 0, the rest zero
        let mut core_b = [[[(0.0, 0.0); 2]; 2]; 2];
        core_b[0][0][0] = b0;
        core_b[0][1][0] = b1;
        let mut core_a = [[[(0.0, 0.0); 2]; 2]; 2];
        core_a[0][0][0] = a0;
        core_a[0][1][0] = a1;

        let out = TensorNetworkUnit::zipper2_step(
            &TensorNetworkUnit::e_boundary(),
            &TensorNetworkUnit::pack_core(&core_b),
            &TensorNetworkUnit::pack_core(&core_a),
        );
        let (got_re, got_im) = TensorNetworkUnit::e_at(&out, 0, 0);

        assert!((got_re as f64 - want_re).abs() < 1e-6, "re: {} vs {}", got_re, want_re);
        assert!((got_im as f64 - want_im).abs() < 1e-6, "im: {} vs {}", got_im, want_im);
    }

    /// GHZ_4 is a genuine bond-dimension-2 state -- the smallest thing chi = 1
    /// provably cannot represent -- so <GHZ|GHZ> = 1 is the honest golden for
    /// this instruction. The head core is (1,2,2) and the tail (2,2,1); both are
    /// zero-padded into the uniform block, which is exactly the case that would
    /// return a plausible wrong number if the padding were wrong.
    ///
    /// Cross-checked against a reference contraction, which gives
    /// 0.999999940395355 -- 1 minus one f32 epsilon, with the cores quantizing
    /// losslessly (GHZ entries are +-1/sqrt(2), the degenerate case
    /// documents in phase 8).
    #[test]
    fn test_zipper2_contracts_ghz4_to_unit_norm() {
        let inv_sqrt2 = 1.0 / (2.0f64).sqrt();

        let mut head = [[[(0.0, 0.0); 2]; 2]; 2];
        head[0][0][0] = (1.0, 0.0);
        head[0][1][1] = (1.0, 0.0);

        let mut mid = [[[(0.0, 0.0); 2]; 2]; 2];
        mid[0][0][0] = (1.0, 0.0);
        mid[1][1][1] = (1.0, 0.0);

        let mut tail = [[[(0.0, 0.0); 2]; 2]; 2];
        tail[0][0][0] = (inv_sqrt2, 0.0);
        tail[1][1][0] = (inv_sqrt2, 0.0);

        let ghz4 = [head, mid, mid, tail];
        let mut e = TensorNetworkUnit::e_boundary();
        for core in ghz4.iter() {
            let packed = TensorNetworkUnit::pack_core(core);
            e = TensorNetworkUnit::zipper2_step(&e, &packed, &packed);
        }

        let (re, im) = TensorNetworkUnit::e_at(&e, 0, 0);
        assert!((re as f64 - 1.0).abs() < 1e-6, "<GHZ|GHZ> = {} + {}i, esperado 1", re, im);
        assert!((im as f64).abs() < 1e-6, "la norma debe ser real, im = {}", im);
    }

    /// The discriminating golden. `<GHZ|GHZ> = 1` above proves the plumbing but
    /// not the capability -- *every* normalized state has norm 1, chi = 1
    /// included -- so it cannot tell a correct chi = 2 contraction from a broken
    /// one that happens to stay normalized.
    ///
    /// `<GHZ_4 | +^4>` can. It is 1/(2*sqrt(2)) = 0.353553390593274, it needs the
    /// off-diagonal entries of the transfer matrix to come out right, and it
    /// drops to a visibly wrong number if the padding or the index order is off
    /// by anything. Agreed to 3.6e-8 against both a reference contraction and
    /// the closed form 2^(1-n/2)/sqrt(2) at n = 4, 6, 8 before being frozen here.
    #[test]
    fn test_zipper2_overlap_of_ghz4_with_the_plus_state() {
        let inv_sqrt2 = 1.0 / (2.0f64).sqrt();

        let mut head = [[[(0.0, 0.0); 2]; 2]; 2];
        head[0][0][0] = (1.0, 0.0);
        head[0][1][1] = (1.0, 0.0);
        let mut mid = [[[(0.0, 0.0); 2]; 2]; 2];
        mid[0][0][0] = (1.0, 0.0);
        mid[1][1][1] = (1.0, 0.0);
        let mut tail = [[[(0.0, 0.0); 2]; 2]; 2];
        tail[0][0][0] = (inv_sqrt2, 0.0);
        tail[1][1][0] = (inv_sqrt2, 0.0);
        let ghz4 = [head, mid, mid, tail];

        // |+>^4: a (1,2,1) core at every site, padded into the uniform block.
        let mut plus = [[[(0.0, 0.0); 2]; 2]; 2];
        plus[0][0][0] = (inv_sqrt2, 0.0);
        plus[0][1][0] = (inv_sqrt2, 0.0);
        let packed_plus = TensorNetworkUnit::pack_core(&plus);

        // The bra is conjugated, so GHZ goes in as `core_a`.
        let mut e = TensorNetworkUnit::e_boundary();
        for core in ghz4.iter() {
            e = TensorNetworkUnit::zipper2_step(
                &e, &packed_plus, &TensorNetworkUnit::pack_core(core));
        }

        let (re, im) = TensorNetworkUnit::e_at(&e, 0, 0);
        let want = 1.0 / (2.0 * (2.0f64).sqrt());
        assert!((re as f64 - want).abs() < 1e-6, "<GHZ_4|+^4> = {}, esperado {}", re, want);
        assert!((im as f64).abs() < 1e-6, "debe ser real, im = {}", im);
    }

    /// The packed core must survive the round trip to within the int8 step, and
    /// the scale must be per right bond -- two bonds of very different magnitude
    /// must both keep their precision, which is the whole reason per-bond was
    /// chosen over a single scale for the register.
    #[test]
    fn test_pack_core_keeps_each_bond_at_full_int8_precision() {
        let mut core = [[[(0.0, 0.0); 2]; 2]; 2];
        core[0][0][0] = (1.0, -0.5);        // bond 0: order 1
        core[1][1][0] = (0.25, 0.75);
        core[0][0][1] = (1.0e-6, -5.0e-7);  // bond 1: six orders smaller
        core[1][1][1] = (2.5e-7, 7.5e-7);

        let packed = TensorNetworkUnit::pack_core(&core);
        let back = TensorNetworkUnit::unpack_core(&packed);

        for l in 0..2 {
            for d in 0..2 {
                for r in 0..2 {
                    let (want_re, want_im) = core[l][d][r];
                    let (got_re, got_im) = back[l][d][r];
                    // Peak of this bond / 127 is the quantization step; allow one.
                    let peak: f64 = (0..2).flat_map(|ll| (0..2).map(move |dd| (ll, dd)))
                        .map(|(ll, dd)| {
                            let (re, im) = core[ll][dd][r];
                            re.abs().max(im.abs())
                        })
                        .fold(0.0f64, f64::max);
                    let step = peak / 127.0;
                    assert!((got_re as f64 - want_re).abs() <= step,
                            "[{}][{}][{}] re {} vs {}", l, d, r, got_re, want_re);
                    assert!((got_im as f64 - want_im).abs() <= step,
                            "[{}][{}][{}] im {} vs {}", l, d, r, got_im, want_im);
                }
            }
        }
    }
}



