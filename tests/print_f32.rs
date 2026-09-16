//! Printing a float, and printing it so it can be read back.
//!
//! The machine has had eight f32 lanes and packed arithmetic over them since the float unit
//! landed, and no way to emit one. `PRINT_F64` takes f64 bits and there is no conversion
//! instruction to make them from an f32, so a program that computed a float could only dump
//! the register as hex.

use unibit::assembler::Assembler;
use unibit::binary::{self, Object};
use unibit::cpu::Cpu;

fn run(src: &str) -> String {
    let prog = Assembler::new().assemble(src).expect("assembles");
    let bytes = binary::write_object(&Object {
        entry_point: prog.entry_point,
        code: prog.instructions,
        data: prog.data_segment,
    });
    let obj = binary::read_object(&bytes).expect("round-trips");
    let mut cpu = Cpu::new(1024 * 1024);
    cpu.capture_output = true;
    for (addr, seg) in &obj.data {
        cpu.memory.write_bytes(*addr, seg).expect("data fits");
    }
    cpu.reset(obj.entry_point);
    cpu.run_program(&obj.code, 100_000).expect("runs");
    String::from_utf8(cpu.stdout_buffer).expect("utf-8")
}

fn print_bits(bits: u32) -> String {
    run(&format!(
        "        .text\n        .global _start\n_start:\n        li      a0, 0x{bits:08X}\n        li      a7, 7\n        ecall\n        halt\n"
    ))
}

#[test]
fn a_float_comes_out_as_a_float() {
    assert_eq!(print_bits(3.5f32.to_bits()), "3.5");
    assert_eq!(print_bits((-0.125f32).to_bits()), "-0.125");
    assert_eq!(print_bits(0f32.to_bits()), "0");
}

#[test]
fn what_it_prints_parses_back_to_the_bits_it_printed() {
    // The property the syscall exists for, and the reason it does not use `{:.6}` the way
    // `PRINT_F64` beside it does. A program's output on this machine is compared against a
    // host oracle bit for bit; a float printed to six places cannot make that trip, because
    // many distinct f32 share the same eight characters.
    //
    // Awkward values on purpose: a repeating decimal, a subnormal, and two neighbours one ulp
    // apart that fixed precision would print identically.
    let awkward: [f32; 6] = [
        0.1,
        1.0 / 3.0,
        f32::from_bits(1),
        1.000_000_1,
        1.000_000_2,
        -1.234_567_8e-7,
    ];
    for v in awkward {
        let text = print_bits(v.to_bits());
        let back: f32 = text
            .parse()
            .unwrap_or_else(|e| panic!("`{text}` does not parse back: {e}"));
        assert_eq!(back.to_bits(), v.to_bits(), "`{text}` is not {v:e}");
    }

    // And the pair really is a pair: if these two printed the same string the test above
    // would be checking nothing.
    assert_ne!(print_bits(1.000_000_1f32.to_bits()), print_bits(1.000_000_2f32.to_bits()));
}

#[test]
fn only_the_low_thirty_two_bits_are_the_float() {
    // `a0` is a 256-bit register and an f32 is 32 bits of it. Reading the whole low lane as
    // f64 bits -- which is what `PRINT_F64` does with the same register -- gives a different
    // number entirely, so the two syscalls are not interchangeable.
    let f64_view = run(
        "        .text\n        .global _start\n_start:\n        li      a0, 0x40600000\n        li      a7, 5\n        ecall\n        halt\n",
    );
    assert_eq!(print_bits(0x4060_0000), "3.5");
    assert_ne!(f64_view, "3.5");
}
