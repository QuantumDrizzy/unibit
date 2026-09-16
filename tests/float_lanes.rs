//! Packed single-precision, end to end: assemble, encode, decode, execute.
//!
//! `Width::B32` has been documented as "8 lanes x 32-bit (AI training, fp32)" since the
//! register model was written, and `Reg256` has carried `f32_at`/`set_f32_at` since the tensor
//! unit needed an accumulator. Until now **eight floats fitted in a register and nothing in
//! the ALU could add them**: `VADD` and friends are integer, `wrapping_add` over lanes, and the
//! only float arithmetic in the machine was the two-lane complex unit.
//!
//! This is the gap closed. The tests below are chosen to fail against the three ways it could
//! be closed wrongly: integer lanes wearing a float name, an unfused `VFMA`, and a round trip
//! through the object format that loses the opcode.

use unibit::alu::FloatUnit;
use unibit::assembler::Assembler;
use unibit::isa::{Instruction, Reg256};

fn packed(v: [f32; 8]) -> Reg256 {
    let mut r = Reg256::ZERO;
    for (i, x) in v.iter().enumerate() {
        r.set_f32_at(i, *x);
    }
    r
}

fn lanes(r: &Reg256) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    for (i, o) in out.iter_mut().enumerate() {
        *o = r.f32_at(i);
    }
    out
}

#[test]
fn eight_lanes_of_real_float_arithmetic() {
    let a = packed([1.5, -2.0, 0.25, 1e10, -0.0, 3.5, 100.0, 0.1]);
    let b = packed([2.5, 0.5, 4.0, 1e10, 0.0, -3.5, 0.5, 0.2]);

    assert_eq!(
        lanes(&FloatUnit::vfadd(&a, &b)),
        [4.0, -1.5, 4.25, 2e10, 0.0, 0.0, 100.5, 0.1f32 + 0.2f32]
    );
    assert_eq!(
        lanes(&FloatUnit::vfmul(&a, &b)),
        [3.75, -1.0, 1.0, 1e20, -0.0, -12.25, 50.0, 0.1f32 * 0.2f32]
    );
    // 0.1 + 0.2 is deliberately in there: a lane that quietly went through f64 would give a
    // different answer from `0.1f32 + 0.2f32`, and the test would say so.
}

#[test]
fn float_lanes_do_not_wrap_they_overflow() {
    // The difference between this unit and `VADD`. An integer lane wraps; a float lane
    // saturates to infinity, and a NaN propagates. Getting this wrong would make the
    // instruction an integer op with a float name -- which still produces numbers.
    let big = packed([f32::MAX; 8]);
    let out = FloatUnit::vfadd(&big, &big);
    assert!(out.f32_at(0).is_infinite(), "f32::MAX + f32::MAX is infinity, not a wrap");
    assert!(out.f32_at(7).is_infinite());

    let nan = packed([f32::NAN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let one = packed([1.0; 8]);
    assert!(FloatUnit::vfmul(&nan, &one).f32_at(0).is_nan(), "NaN must propagate");
    assert_eq!(FloatUnit::vfmul(&nan, &one).f32_at(1), 1.0, "and not spread");
}

#[test]
fn vfma_is_fused_which_is_a_different_answer() {
    // The test this instruction exists to pass. `a * b + c` rounded twice is not `a * b + c`
    // rounded once, and an ISA that offers VFMA and implements it as a multiply followed by an
    // add is lying about what it computes.
    //
    // Choosing values that separate the two is the whole difficulty, and the first attempt
    // here did not: `(1+e)^2 - 1` gives `2e` both ways, because `e^2` lands exactly on a tie
    // and rounds away in the fused form too. A test where the two agree cannot tell them
    // apart, which makes it worse than no test.
    //
    // `0.1 * 10 - 1` does separate them. `0.1f32` is 0.100000001490116..., so the exact
    // product is 1.0000000149...: rounding it to f32 first snaps it to 1.0 and the subtraction
    // gives **zero**, while keeping it and subtracting once gives the 1.49e-8 that was really
    // there.
    let a = packed([0.1; 8]);
    let b = packed([10.0; 8]);
    let c = packed([-1.0; 8]);

    let fused = FloatUnit::vfma(&a, &b, &c);
    let unfused: Vec<f32> = (0..8).map(|i| a.f32_at(i) * b.f32_at(i) + c.f32_at(i)).collect();

    assert_eq!(unfused[0], 0.0, "rounded twice, the difference vanishes");
    assert_ne!(
        fused.f32_at(0),
        unfused[0],
        "if these agree the instruction is not fused and the test cannot tell"
    );
    assert_eq!(fused.f32_at(0), 0.1f32.mul_add(10.0, -1.0));
    assert!(fused.f32_at(0) > 0.0, "and it is the residue that was actually there");
}

#[test]
fn max_and_min_follow_ieee_on_nan() {
    // `f32::max` returns the non-NaN operand, which is what IEEE 754 says and is not what a
    // naive `if a > b` comparison does -- that would propagate the NaN.
    let nan = packed([f32::NAN; 8]);
    let ones = packed([1.0; 8]);
    assert_eq!(FloatUnit::vfmax(&nan, &ones).f32_at(0), 1.0);
    assert_eq!(FloatUnit::vfmin(&nan, &ones).f32_at(0), 1.0);
    assert_eq!(FloatUnit::vfmax(&ones, &packed([2.0; 8])).f32_at(3), 2.0);
    assert_eq!(FloatUnit::vfmin(&ones, &packed([2.0; 8])).f32_at(3), 1.0);
}

#[test]
fn the_assembler_accepts_them_without_a_width_suffix() {
    // f32 is 32 bits, so a width suffix would have exactly one legal value -- and a field with
    // one legal value is a field somebody will eventually set to the other one.
    let src = "\
        .text
        .global _start
_start:
        vfadd   t0, t1, t2
        vfsub   t0, t1, t2
        vfmul   t0, t1, t2
        vfma    t0, t1, t2
        vfmax   t0, t1, t2
        vfmin   t0, t1, t2
        vfreduce t0, t1
        halt
";
    let prog = Assembler::new()
        .assemble(src)
        .expect("the seven mnemonics assemble");
    let names: Vec<&str> = prog
        .instructions
        .iter()
        .map(|i| match i {
            Instruction::VfAdd { .. } => "vfadd",
            Instruction::VfSub { .. } => "vfsub",
            Instruction::VfMul { .. } => "vfmul",
            Instruction::VfMa { .. } => "vfma",
            Instruction::VfMax { .. } => "vfmax",
            Instruction::VfMin { .. } => "vfmin",
            Instruction::VfReduce { .. } => "vfreduce",
            _ => "other",
        })
        .collect();
    assert_eq!(
        &names[..7],
        &["vfadd", "vfsub", "vfmul", "vfma", "vfmax", "vfmin", "vfreduce"]
    );
}

/// Sum the lanes left to right, the order `VREDUCE` uses for integers.
fn sequential(v: [f32; 8]) -> f32 {
    v.iter().fold(0.0f32, |a, x| a + x)
}

/// Sum the lanes in a tree: stride 4, then 2, then 1.
fn tree(v: [f32; 8]) -> f32 {
    let mut s = v;
    let mut stride = 4;
    while stride > 0 {
        for t in 0..stride {
            s[t] += s[t + stride];
        }
        stride /= 2;
    }
    s[0]
}

/// One large value and seven at half an ulp of it. Each small one is lost against the large
/// one on its own, and two of them together are not -- so the two orders disagree by three
/// ulps rather than by a last bit.
const OBSERVABLE: [f32; 8] = [
    1.0,
    5.960_464_5e-8,
    5.960_464_5e-8,
    5.960_464_5e-8,
    5.960_464_5e-8,
    5.960_464_5e-8,
    5.960_464_5e-8,
    5.960_464_5e-8,
];

#[test]
fn the_horizontal_sum_is_a_tree_and_that_is_observable() {
    // The test would pass vacuously on any vector where the two orders agree, which is most of
    // them, so it asserts that they disagree *first*.
    assert_ne!(
        sequential(OBSERVABLE).to_bits(),
        tree(OBSERVABLE).to_bits(),
        "this vector cannot tell the two orders apart, so it cannot test the choice"
    );
    assert_eq!(sequential(OBSERVABLE).to_bits(), 0x3F80_0000);
    assert_eq!(tree(OBSERVABLE).to_bits(), 0x3F80_0003);

    let got = FloatUnit::vfreduce(&packed(OBSERVABLE));
    assert_eq!(
        got.f32_at(0).to_bits(),
        tree(OBSERVABLE).to_bits(),
        "VFREDUCE is specified as a tree"
    );
}

#[test]
fn the_integer_reduce_never_had_that_choice_to_make() {
    // Why `VFREDUCE` documents its order and `VREDUCE` beside it does not. `wrapping_add` is
    // associative, so no vector of integers can tell a chain from a tree -- the order was
    // never observable and so was never a decision. Float addition is not associative, and
    // the instruction above had to pick.
    let nasty: [u32; 8] = [
        u32::MAX,
        1,
        u32::MAX / 2,
        7,
        0x8000_0000,
        0xDEAD_BEEF,
        3,
        0xFFFF_FFFE,
    ];
    let seq = nasty.iter().fold(0u64, |a, x| a.wrapping_add(*x as u64));
    let mut s: Vec<u64> = nasty.iter().map(|x| *x as u64).collect();
    let mut stride = 4;
    while stride > 0 {
        for t in 0..stride {
            s[t] = s[t].wrapping_add(s[t + stride]);
        }
        stride /= 2;
    }
    assert_eq!(seq, s[0], "integer lanes cannot distinguish the two orders");
}

#[test]
fn only_lane_zero_carries_the_result() {
    // Stated rather than left undefined. An undefined lane is one a program will eventually
    // read, and it would read differently on the next implementation.
    let r = FloatUnit::vfreduce(&packed([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]));
    assert_eq!(lanes(&r), [255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn it_reduces_through_the_whole_machine() {
    // Assembler, encoder, decoder and CPU, not just the ALU -- the path a LYTH program takes.
    use unibit::binary::{self, Object};
    use unibit::cpu::Cpu;

    let src = "        .text
        .global _start
_start:
        li      t0, 0x40400000
        vsplat.w t0, t0
        vfreduce t1, t0
        halt
";
    let prog = Assembler::new().assemble(src).expect("assembles");
    let bytes = binary::write_object(&Object {
        entry_point: prog.entry_point,
        code: prog.instructions,
        data: prog.data_segment,
    });
    let obj = binary::read_object(&bytes).expect("round-trips");

    let mut cpu = Cpu::new(1024 * 1024);
    cpu.reset(obj.entry_point);
    cpu.run_program(&obj.code, 1_000).expect("runs");

    // Eight lanes of 3.0. The sum is exact in either order, which is what this test wants:
    // it is checking the plumbing, and `the_horizontal_sum_is_a_tree` checks the order.
    assert_eq!(cpu.get_reg(6).f32_at(0), 24.0);
}
