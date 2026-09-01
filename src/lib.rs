// ============================================================================
// FORJA-256 — The Post-Quantum AI-Native 256-bit Architecture
// ============================================================================
//
// A clean, zero-dependency, bare-metal processor engine in Rust.
//
// Modules:
//   - isa: 256-bit register structures & full instruction definitions
//   - memory: Byte-addressable little-endian memory subsystem
//   - alu: Scalar, SIMD Vector, Complex, Info-Theory, PQC Lattice, Tensor Units
//   - cpu: Core pipeline, BHT/BTB predictor, Landauer thermodynamic tracking
//   - assembler: Two-pass assembler, label linker, directives parser
//
// ============================================================================

pub mod isa;
pub mod memory;
pub mod alu;
pub mod cpu;
pub mod assembler;
pub mod disasm;

pub use isa::{Reg256, Instruction, Width, ActivationFn, landauer_energy};
pub use memory::Memory;
pub use cpu::{Cpu, CpuStatus, CpuMetrics};
pub use assembler::{Assembler, AssembledProgram};
pub use disasm::Disassembler;
