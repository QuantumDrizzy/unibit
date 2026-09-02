// ============================================================================
// Unibit — End-to-end integration tests
// ============================================================================
//
// The unit tests cover execution units in isolation. These cover the whole
// pipeline the CLI actually uses:
//
//     .uasm source -> assembler -> UBIT object -> decode -> CPU -> stdout
//
// Every shipped program is executed here, so a regression in any stage fails
// the build rather than showing up as garbled output at demo time.

use std::collections::HashMap;

use unibit::assembler::Assembler;
use unibit::binary::{self, Object};
use unibit::cpu::Cpu;
use unibit::disasm::Disassembler;

const PROGRAMS: &[&str] = &[
    "fibonacci",
    "mandelbrot",
    "quantum_pqc",
    "tensor_network_zipper",
    "master_suite",
    "mps_ghz_overlap",
];

/// Assemble a shipped program from `programs/`.
fn assemble(name: &str) -> unibit::assembler::AssembledProgram {
    let path = format!("programs/{}.uasm", name);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    Assembler::new()
        .assemble(&source)
        .unwrap_or_else(|e| panic!("{} failed to assemble: {}", path, e))
}

/// Run an object to completion with output captured, exactly as the CLI does.
fn run_object(obj: &Object) -> String {
    let mut cpu = Cpu::new(1024 * 1024);
    cpu.capture_output = true;
    for (addr, seg) in &obj.data {
        cpu.memory.write_bytes(*addr, seg).expect("data segment fits in memory");
    }
    cpu.reset(obj.entry_point);
    cpu.run_program(&obj.code, 10_000_000).expect("program ran without trapping");
    String::from_utf8(cpu.stdout_buffer).expect("output is valid UTF-8")
}

/// The full CLI path: assemble, serialise, parse back, execute.
fn build_and_run(name: &str) -> String {
    let program = assemble(name);
    let bytes = binary::write_object(&Object {
        entry_point: program.entry_point,
        code: program.instructions,
        data: program.data_segment,
    });
    let obj = binary::read_object(&bytes)
        .unwrap_or_else(|e| panic!("{} object did not parse back: {}", name, e));
    run_object(&obj)
}

#[test]
fn every_shipped_program_assembles_and_runs() {
    for name in PROGRAMS {
        let out = build_and_run(name);
        assert!(!out.is_empty(), "{} produced no output", name);
    }
}

#[test]
fn object_round_trip_is_lossless_for_every_program() {
    // Serialising and parsing back must not perturb a single instruction.
    for name in PROGRAMS {
        let program = assemble(name);
        let original = program.instructions.clone();
        let bytes = binary::write_object(&Object {
            entry_point: program.entry_point,
            code: program.instructions,
            data: program.data_segment,
        });
        let obj = binary::read_object(&bytes).expect("parses back");
        assert_eq!(obj.code, original, "{}: instructions changed through the object format", name);
    }
}

#[test]
fn encoded_and_direct_execution_agree() {
    // Running the decoded object must produce byte-identical output to running
    // the assembler's instructions directly.
    for name in PROGRAMS {
        let program = assemble(name);
        let direct = run_object(&Object {
            entry_point: program.entry_point,
            code: program.instructions.clone(),
            data: program.data_segment.clone(),
        });
        let via_bytes = build_and_run(name);
        assert_eq!(direct, via_bytes, "{}: output differs after an encode/decode cycle", name);
    }
}

#[test]
fn fibonacci_computes_the_real_sequence() {
    let out = build_and_run("fibonacci");
    for n in ["0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144", "17711", "121393"] {
        assert!(out.contains(n), "fibonacci output missing {:?}:\n{}", n, out);
    }
}

#[test]
fn mandelbrot_renders_a_plausible_set() {
    let out = build_and_run("mandelbrot");
    let rows: Vec<&str> = out.lines().filter(|l| l.contains('#')).collect();
    assert!(rows.len() >= 20, "expected >= 20 rendered rows, got {}", rows.len());

    // The set is symmetric about the real axis: the first and last rendered
    // rows must have the same number of filled cells.
    let filled = |s: &str| s.chars().filter(|c| *c == '#').count();
    assert_eq!(
        filled(rows[0]),
        filled(rows[rows.len() - 1]),
        "render is not symmetric about the real axis"
    );
    // The widest row should be near the middle of the image.
    let widest = rows.iter().enumerate().max_by_key(|(_, r)| filled(r)).unwrap().0;
    let middle = rows.len() / 2;
    assert!(
        widest.abs_diff(middle) <= rows.len() / 4,
        "widest row {} is far from the middle {}",
        widest,
        middle
    );
}

#[test]
fn pqc_program_reports_a_verified_ntt_round_trip() {
    let out = build_and_run("quantum_pqc");
    assert!(
        out.contains("VERIFIED"),
        "the NTT round-trip check did not pass:\n{}",
        out
    );
    assert!(!out.contains("FAILED"), "NTT round-trip reported a failure:\n{}", out);
}

#[test]
fn non_ascii_strings_survive_assembly() {
    // Regression: the assembler used to truncate every char to its low byte,
    // turning box-drawing U+2554 into 'T'.
    let out = build_and_run("quantum_pqc");
    assert!(out.contains('╔'), "box-drawing characters were mangled:\n{}", out);
    assert!(out.contains('✓'), "the checkmark was mangled:\n{}", out);
}

#[test]
fn branch_predictor_learns_a_hot_loop() {
    // Regression: the BHT was indexed by the destination PC, so it never
    // learned and Mandelbrot predicted at ~3%.
    let program = assemble("mandelbrot");
    let mut cpu = Cpu::new(1024 * 1024);
    cpu.capture_output = true;
    for (addr, seg) in &program.data_segment {
        cpu.memory.write_bytes(*addr, seg).unwrap();
    }
    cpu.reset(program.entry_point);
    cpu.run_program(&program.instructions, 10_000_000).unwrap();

    let accuracy = cpu.metrics.branch_accuracy();
    assert!(cpu.metrics.branch_count > 1000, "expected a branch-heavy program");
    assert!(accuracy > 90.0, "branch accuracy regressed to {:.2}%", accuracy);
}

#[test]
fn memory_traffic_is_accounted_for() {
    // Every program prints strings, so the read counter must move; the report
    // used to show 0/0 unconditionally.
    let program = assemble("fibonacci");
    let mut cpu = Cpu::new(1024 * 1024);
    cpu.capture_output = true;
    for (addr, seg) in &program.data_segment {
        cpu.memory.write_bytes(*addr, seg).unwrap();
    }
    cpu.reset(program.entry_point);
    cpu.run_program(&program.instructions, 10_000_000).unwrap();

    assert!(cpu.memory.reads > 0, "string reads were not counted on the bus");
    // Loading the data segment must not inflate the Landauer erasure count.
    assert_eq!(cpu.memory.bit_erasures, 0, "the loader was charged for erasures");
}

#[test]
fn disassembly_works_from_bytes_alone() {
    // No AST, no symbol table: the disassembler must decode the byte stream.
    let program = assemble("fibonacci");
    let expected = program.instructions.len();
    let bytes = binary::write_object(&Object {
        entry_point: program.entry_point,
        code: program.instructions,
        data: program.data_segment,
    });

    let text = Disassembler::disassemble_object(&bytes).expect("disassembles");
    assert!(text.contains("ecall"), "no ecall in the disassembly:\n{}", text);

    // Count only the .text listing: the .data summary lines also start with [0x.
    let code_listing = text.split("\n        .data\n").next().unwrap();
    let decoded = code_listing
        .lines()
        .filter(|l| l.trim_start().starts_with("[0x"))
        .count();
    assert_eq!(decoded, expected, "disassembly lost instructions");
}

#[test]
fn disassembly_matches_the_assembler_listing() {
    // Decoding from bytes must reproduce what formatting the AST produces.
    let program = assemble("master_suite");
    let from_ast = Disassembler::disassemble_instructions(&program.instructions, &HashMap::new());
    let bytes = binary::write_object(&Object {
        entry_point: program.entry_point,
        code: program.instructions,
        data: program.data_segment,
    });
    let obj = binary::read_object(&bytes).unwrap();
    let from_bytes = Disassembler::disassemble_instructions(&obj.code, &HashMap::new());
    assert_eq!(from_ast, from_bytes);
}

#[test]
fn entropy_profile_separates_code_from_text() {
    // An ASCII data section is denser than the sparse fixed-width code
    // records, which is the whole point of scanning for entropy anomalies.
    let program = assemble("mandelbrot");
    let bytes = binary::write_object(&Object {
        entry_point: program.entry_point,
        code: program.instructions,
        data: program.data_segment,
    });
    let profile = Disassembler::scan_entropy_profile(&bytes, 256);
    assert!(profile.len() >= 2, "expected several blocks, got {}", profile.len());
    for (offset, bits) in &profile {
        assert!(
            *bits >= 0.0 && *bits <= 8.0,
            "block at 0x{:x} reports {} bits/byte, outside [0, 8]",
            offset,
            bits
        );
    }
    let last = profile.last().unwrap().1;
    let first = profile[0].1;
    assert!(last > first, "the text section should be denser than the code section");
}

#[test]
fn mps_program_reproduces_the_analytic_overlap() {
    // <GHZ'_4|i+^4> = (1-i)/(4*sqrt2), contracted at bond dimension 2 through
    // the ZIPPER2 instruction and checked in-band by the program itself.
    let out = build_and_run("mps_ghz_overlap");
    assert!(out.contains("[OK]"), "the chi=2 overlap check did not pass:\n{}", out);
    assert!(!out.contains("FAIL"), "the chi=2 overlap check failed:\n{}", out);
}

#[test]
fn assembler_rejects_broken_sources() {
    let cases: &[(&str, &str)] = &[
        ("unknown mnemonic", "        .text\n_start:\n        frobnicate t0, t1\n"),
        ("unknown label", "        .text\n_start:\n        beq t0, zero, nowhere\n"),
        ("unknown register", "        .text\n_start:\n        add q9, t1, t2\n"),
        ("unresolved la", "        .text\n_start:\n        la a0, missing_symbol\n"),
    ];
    for (what, src) in cases {
        assert!(
            Assembler::new().assemble(src).is_err(),
            "assembler accepted a source with {}",
            what
        );
    }
}

#[test]
fn object_parser_rejects_corrupt_input() {
    let program = assemble("fibonacci");
    let good = binary::write_object(&Object {
        entry_point: program.entry_point,
        code: program.instructions,
        data: program.data_segment,
    });

    let mut bad_magic = good.clone();
    bad_magic[2] = b'X';
    assert!(binary::read_object(&bad_magic).is_err(), "accepted bad magic");

    let mut bad_version = good.clone();
    bad_version[4] = 99;
    assert!(binary::read_object(&bad_version).is_err(), "accepted an unknown version");

    let mut bad_opcode = good.clone();
    bad_opcode[32] = 0xFD;
    assert!(binary::read_object(&bad_opcode).is_err(), "accepted an unknown opcode");

    assert!(binary::read_object(&good[..good.len() / 2]).is_err(), "accepted a truncated file");
    assert!(binary::read_object(b"").is_err(), "accepted an empty file");
}

#[test]
fn execution_is_deterministic() {
    // Same input, same output: no clock, address or hash-order dependence.
    for name in PROGRAMS {
        let a = build_and_run(name);
        let b = build_and_run(name);
        assert_eq!(a, b, "{} is not deterministic across runs", name);
    }
}
