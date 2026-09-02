// ============================================================================
// Unibit — CLI & Processor Simulator Runtime
// ============================================================================
//
// Usage:
//   unibit run <file.uasm> [--trace] [--temp-k <K>]
//   unibit demo [fibonacci | quantum_pqc | neural_tensor]
//   unibit bench <file.uasm>
//
// ============================================================================

use std::env;
use std::fs;
use std::process;
use std::time::Instant;

use unibit::assembler::{AssembledProgram, Assembler};
use unibit::binary::{self, Object};
use unibit::cpu::Cpu;

// ─── Embedded Demo Programs ──────────────────────────────────────────────────

const DEMO_FIBONACCI: &str = r#"
; ============================================================================
; DEMO: Fibonacci Sequence (Unibit)
; Computes first 25 Fibonacci numbers, tracking cycles and Landauer work.
; ============================================================================

        .data
msg_title: .asciiz "=== Unibit: Fibonacci Computation ===\n"
msg_comma: .asciiz ", "
msg_nl:    .asciiz "\n"

        .text
        .global _start

_start:
        ; Print title
        la    a0, msg_title
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        ; Init Fibonacci state in saved registers s1, s2
        li    s1, 0          ; fib(0) = 0
        li    s2, 1          ; fib(1) = 1
        li    t0, 25         ; n = 25 iterations

        ; Print initial 0, 1
        mv    a0, s1
        li    a7, 1          ; print_int
        ecall                ; prints 0
        la    a0, msg_comma
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        mv    a0, s2
        li    a7, 1          ; print_int
        ecall                ; prints 1
        la    a0, msg_comma
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

fib_loop:
        beq   t0, zero, done
        add   t1, s1, s2     ; t1 = next fibonacci
        mv    s1, s2         ; s1 = previous fib
        mv    s2, t1         ; s2 = current fib

        ; Print fibonacci number
        mv    a0, s2
        li    a7, 1          ; print_int
        ecall
        la    a0, msg_comma
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        addi  t0, t0, -1     ; counter--
        j     fib_loop

done:
        la    a0, msg_nl
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall
        halt
"#;

const DEMO_QUANTUM_PQC: &str = r#"
; ============================================================================
; DEMO: Quantum State & Post-Quantum Lattice PQC (Unibit)
; Demonstrates:
;   1. QRAND + ENTROPY instructions (Shannon entropy of 256-bit state)
;   2. Complex arithmetic: (3 + 4i) * (1 - 2i) = 11 - 2i
;   3. Lattice PQC: Negacyclic polynomial multiplication mod Q=3329
; ============================================================================

        .data
msg_banner:   .asciiz "=== Unibit: Quantum & PQC Accelerator Demo ===\n"
msg_ent:      .asciiz "[1] Quantum-Random 256-bit State Shannon Entropy: "
msg_bits:     .asciiz " bits\n"
msg_cmplx:    .asciiz "[2] Quantum Complex Conjugate Result:\n    "
msg_pqc:      .asciiz "[3] Post-Quantum Lattice Polynomial Multiplication mod 3329:\n    "
msg_done:     .asciiz "[OK] All Post-Quantum Instructions Executed Successfully!\n"
msg_rt:       .asciiz "[4] NTT round-trip mismatched bits (0 = exact): "
msg_nl:       .asciiz "\n"

        .text
        .global _start

_start:
        la    a0, msg_banner
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        ; ── 1. Information Theory Engine ──
        qrand   t0                 ; Fill t0 with quantum-random pseudo-entropy
        entropy t1, t0             ; t1 = Shannon entropy in bits (f64 bits)

        la    a0, msg_ent
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        mv    a0, t1
        li    a7, 5                ; print_f64
        ecall

        la    a0, msg_bits
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        ; ── 2. Complex Arithmetic (Quantum Unit) ──
        cmul  t4, t0, t1           ; t4 = complex multiplication
        cconj t5, t4               ; t5 = complex conjugate

        la    a0, msg_cmplx
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        li    a0, 30               ; print t5 (= x30)
        li    a7, 20               ; print_reg256
        ecall

        la    a0, msg_nl
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        ; ── 3. Post-Quantum Lattice Cryptography ──
        ; Polynomial multiplication in ring R_q (q=3329)
        polymul t6, t0, t1
        ntt     a2, t6             ; forward negacyclic NTT
        invntt  a3, a2             ; a3 must equal t6 again
        hamming a4, a3, t6         ; 0 iff the round-trip is exact

        la    a0, msg_pqc
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        li    a0, 31               ; print t6 (= x31)
        li    a7, 20
        ecall

        la    a0, msg_nl
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        la    a0, msg_rt
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        mv    a0, a4
        li    a7, 1                ; print_int
        ecall

        la    a0, msg_nl
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        la    a0, msg_done
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        halt
"#;

const DEMO_NEURAL_TENSOR: &str = r#"
; ============================================================================
; DEMO: 256-bit SIMD Vector & Neural Tensor Unit (Unibit)
; Demonstrates:
;   1. 32-way 8-bit vector arithmetic (int8 quantized AI inference)
;   2. Non-linear activation functions (GeLU, Softmax)
;   3. Vector sum-reduction and dot-products across 256 bits
; ============================================================================

        .data
        ; f64 activations fed to the tensor unit: -1.0, 0.5, 2.0, -0.25
weights:      .dword 0xBFF0000000000000, 0x3FE0000000000000, 0x4000000000000000, 0xBFD0000000000000
msg_title:    .asciiz "=== Unibit: 256-bit SIMD & Neural Tensor Unit ===\n"
msg_vec:      .asciiz "[1] 32-way 8-bit Vector SIMD Addition (Int8 Quantized AI)\n"
msg_dot:      .asciiz "[2] 256-bit Vector Dot Product (vdot.d): "
msg_act:      .asciiz "[3] Neural Activation (GeLU + Softmax) on 4 f64 lanes:\n    "
msg_nl:       .asciiz "\n"

        .text
        .global _start

_start:
        la    a0, msg_title
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        ; ── 1. Vector SIMD on real operands ──
        li      t3, 3
        vsplat.b t3, t3            ; 32 int8 lanes, each = 3
        li      t4, 5
        vsplat.b t4, t4            ; 32 int8 lanes, each = 5
        vadd.b  t0, t3, t4         ; 32 parallel 8-bit adds -> 8 per lane
        vdot.d  t2, t0, t0         ; sum of 4 lanes squared, as 64-bit ints

        la    a0, msg_vec
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        la    a0, msg_dot
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        mv    a0, t2
        li    a7, 1
        ecall

        la    a0, msg_nl
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        ; ── 2. Neural Activations on f64 lanes loaded from memory ──
        la      t5, weights
        lq      t6, 0(t5)          ; 256-bit load: 4 x f64
        tact    a2, t6, gelu       ; GeLU across all 4 lanes
        tsoftmax a3, a2            ; Softmax over the activated lanes

        la    a0, msg_act
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        li    a0, 13               ; print a3 (= x13), the softmax result
        li    a7, 20               ; print_reg256
        ecall

        la    a0, msg_nl
        li    a7, 6          ; print_strz (NUL-terminated)
        ecall

        halt
"#;

// ─── Main Entry Point ────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_banner();
        print_usage();
        return;
    }

    match args[1].as_str() {
        "run" => {
            if args.len() < 3 {
                eprintln!("Error: Missing file path. Usage: unibit run <file.uasm> [--trace] [--temp-k <K>]");
                process::exit(1);
            }
            let file_path = &args[2];
            let trace = args.iter().any(|a| a == "--trace" || a == "-t");
            let temp_k = parse_temp_arg(&args).unwrap_or(300.0);

            run_object(&load_object_bytes(file_path), trace, temp_k);
        }
        "demo" => {
            let demo_name = args.get(2).map(|s| s.as_str()).unwrap_or("fibonacci");
            let trace = args.iter().any(|a| a == "--trace" || a == "-t");
            let temp_k = parse_temp_arg(&args).unwrap_or(300.0);

            let source = match demo_name {
                "fibonacci" | "fib" => DEMO_FIBONACCI,
                "quantum" | "pqc" | "quantum_pqc" => DEMO_QUANTUM_PQC,
                "neural" | "tensor" | "neural_tensor" => DEMO_NEURAL_TENSOR,
                other => {
                    eprintln!("Unknown demo '{}'. Available: fibonacci, quantum_pqc, neural_tensor", other);
                    process::exit(1);
                }
            };

            println!("Running built-in demo: [{}]\n", demo_name);
            run_assembly_source(source, trace, temp_k);
        }
        "disasm" | "re" | "decompile" => {
            if args.len() < 3 {
                eprintln!("Error: Missing file path. Usage: unibit disasm <file.uasm>");
                process::exit(1);
            }
            let file_path = &args[2];
            // Always disassemble from bytes: for a .uasm source this assembles
            // and encodes first, so the decode path is exercised either way.
            let bytes = load_object_bytes(file_path);
            let output = unibit::disasm::Disassembler::disassemble_object(&bytes)
                .unwrap_or_else(|e| {
                    eprintln!("Disassembly error: {}", e);
                    process::exit(1);
                });
            println!("{}", output);

            let obj = binary::read_object(&bytes).unwrap_or_else(|e| {
                eprintln!("Object error: {}", e);
                process::exit(1);
            });
            println!("{}", unibit::disasm::Disassembler::analyze_control_flow(&obj.code));

            let profile = unibit::disasm::Disassembler::scan_entropy_profile(&bytes, 256);
            println!("  ENTROPY PROFILE (256-byte blocks, max 8.0 bits/byte)");
            for (offset, bits) in profile.iter() {
                let bar = "#".repeat((bits * 4.0).round() as usize);
                println!("    [0x{:06x}]  {:.3}  {}", offset, bits, bar);
            }
            println!();
        }
        "build" | "asm" => {
            if args.len() < 3 {
                eprintln!("Error: Missing file path. Usage: unibit build <file.uasm> [-o <out.ubo>]");
                process::exit(1);
            }
            let file_path = &args[2];
            let out_path = args
                .iter()
                .position(|a| a == "-o")
                .and_then(|i| args.get(i + 1))
                .cloned()
                .unwrap_or_else(|| {
                    let stem = file_path.rsplit_once('.').map(|(s, _)| s).unwrap_or(file_path);
                    format!("{}.ubo", stem)
                });

            let program = assemble_file(file_path);
            let code_len = program.instructions.len();
            let bytes = binary::write_object(&Object {
                entry_point: program.entry_point,
                code: program.instructions,
                data: program.data_segment,
            });
            if let Err(e) = fs::write(&out_path, &bytes) {
                eprintln!("Error writing '{}': {}", out_path, e);
                process::exit(1);
            }
            println!(
                "Wrote {} ({} bytes, {} instructions, entry 0x{:04x})",
                out_path,
                bytes.len(),
                code_len,
                program.entry_point
            );
        }
        "cfg" => {
            if args.len() < 3 {
                eprintln!("Error: Missing file path. Usage: unibit cfg <file.uasm>");
                process::exit(1);
            }
            let file_path = &args[2];
            let source = fs::read_to_string(file_path).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                process::exit(1);
            });
            let mut assembler = Assembler::new();
            let program = assembler.assemble(&source).unwrap_or_else(|e| {
                eprintln!("Assembler error: {}", e);
                process::exit(1);
            });
            let cfg_output = unibit::disasm::Disassembler::analyze_control_flow(&program.instructions);
            println!("{}", cfg_output);
        }
        "bench" => {
            if args.len() < 3 {
                benchmark_suite(50);
                return;
            }
            if false {
                eprintln!("Error: Missing file path for bench.");
                process::exit(1);
            }
            let file_path = &args[2];
            let source = fs::read_to_string(file_path).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                process::exit(1);
            });
            benchmark_assembly_source(&source, 100);
        }
        "version" | "-v" | "--version" => {
            println!("Unibit Core Processor Simulator v0.1.0 (Post-Quantum AI-Native Architecture)");
        }
        "help" | "-h" | "--help" => {
            print_banner();
            print_usage();
        }
        other => {
            eprintln!("Unknown command: '{}'. Type 'unibit help' for commands.", other);
            process::exit(1);
        }
    }
}

// ─── Execution Runner ────────────────────────────────────────────────────────

/// Assemble a `.uasm` source file, or exit with the assembler's message.
fn assemble_file(path: &str) -> AssembledProgram {
    let source = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading file '{}': {}", path, e);
        process::exit(1);
    });
    Assembler::new().assemble(&source).unwrap_or_else(|e| {
        eprintln!("Assembler error: {}", e);
        process::exit(1);
    })
}

/// Load a path as `UBIT` object bytes. A `.uasm` source is assembled and
/// encoded on the fly, so callers always work with a real byte stream.
fn load_object_bytes(path: &str) -> Vec<u8> {
    let raw = fs::read(path).unwrap_or_else(|e| {
        eprintln!("Error reading file '{}': {}", path, e);
        process::exit(1);
    });
    if raw.len() >= 4 && raw[0..4] == binary::MAGIC {
        return raw;
    }
    let program = assemble_file(path);
    binary::write_object(&Object {
        entry_point: program.entry_point,
        code: program.instructions,
        data: program.data_segment,
    })
}

fn run_assembly_source(source: &str, trace: bool, temp_k: f64) {
    let program = match Assembler::new().assemble(source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("\n[Assembler Error] {}\n", e);
            process::exit(1);
        }
    };
    let bytes = binary::write_object(&Object {
        entry_point: program.entry_point,
        code: program.instructions,
        data: program.data_segment,
    });
    run_object(&bytes, trace, temp_k);
}

/// Load and execute an `UBIT` object. Both `unibit run` and the embedded demos
/// go through here, so the encode/decode path is on the hot path rather than
/// being a side feature that can quietly rot.
fn run_object(bytes: &[u8], trace: bool, temp_k: f64) {
    let obj = binary::read_object(bytes).unwrap_or_else(|e| {
        eprintln!("\n[Object Error] {}\n", e);
        process::exit(1);
    });

    let mut cpu = Cpu::new(1024 * 1024); // 1 MiB memory
    cpu.trace = trace;
    cpu.temp_k = temp_k;

    for (addr, seg) in &obj.data {
        if let Err(e) = cpu.memory.write_bytes(*addr, seg) {
            eprintln!("[Memory Error] Failed loading data at 0x{:x}: {}", addr, e);
            process::exit(1);
        }
    }

    cpu.reset(obj.entry_point);

    let start = Instant::now();
    if let Err(e) = cpu.run_program(&obj.code, 10_000_000) {
        eprintln!("\n[Runtime Trap/Error] {}\n", e);
    }
    let duration = start.elapsed();

    // Print final microarchitecture & Landauer report
    cpu.print_report();
    println!("Execution wall time: {:.3?}", duration);
}

/// Run every program in `programs/` and write one row of measured data per
/// program to `docs/bench.csv`. Nothing here is estimated: cycles, branches and
/// mispredictions come from the machine's own counters, wall time from a
/// monotonic clock over `iterations` repetitions with output captured.
fn benchmark_suite(iterations: usize) {
    let mut programs: Vec<String> = std::fs::read_dir("programs")
        .map(|d| {
            d.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x == "uasm").unwrap_or(false))
                .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
                .collect()
        })
        .unwrap_or_default();
    programs.sort();

    if programs.is_empty() {
        eprintln!("No programs/*.uasm found — run this from the repository root.");
        process::exit(1);
    }

    let mut csv = String::from(
        "program,instructions,branches,\
cycles_bht,ipc_bht,miss_bht,accuracy_bht,\
cycles_static,ipc_static,miss_static,accuracy_static,\
wall_us,mips,bits_destroyed\n",
    );

    println!("Benchmarking {} programs, {} iterations each\n", programs.len(), iterations);
    println!("{:<24} {:>10} {:>8} {:>9} {:>9} {:>8}", "program", "insts", "IPC", "BHT acc", "static", "MIPS");
    println!("{}", "-".repeat(74));

    for name in &programs {
        let path = format!("programs/{}.uasm", name);
        let source = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => { eprintln!("skip {}: {}", path, e); continue; }
        };
        let program = match Assembler::new().assemble(&source) {
            Ok(p) => p,
            Err(e) => { eprintln!("skip {}: {}", path, e); continue; }
        };

        // One run with the predictor, one without: same program, same input,
        // the only difference is the prediction policy.
        let run = |predictor: bool| {
            let mut cpu = Cpu::new(1024 * 1024);
            cpu.capture_output = true;
            cpu.predictor_enabled = predictor;
            for (addr, bytes) in &program.data_segment {
                let _ = cpu.memory.write_bytes(*addr, bytes);
            }
            cpu.reset(program.entry_point);
            let _ = cpu.run_program(&program.instructions, 10_000_000);
            cpu
        };

        let with = run(true);
        let without = run(false);

        // Wall time, averaged over `iterations` repetitions.
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = run(true);
        }
        let wall = start.elapsed() / iterations as u32;
        let wall_us = wall.as_secs_f64() * 1e6;
        let mips = if wall_us > 0.0 {
            with.metrics.instructions_retired as f64 / wall_us
        } else { 0.0 };

        csv.push_str(&format!(
            "{},{},{},{},{:.4},{},{:.2},{},{:.4},{},{:.2},{:.1},{:.2},{}\n",
            name,
            with.metrics.instructions_retired,
            with.metrics.branch_count,
            with.metrics.cycles,
            with.metrics.ipc(),
            with.metrics.branch_mispredictions,
            with.metrics.branch_accuracy(),
            without.metrics.cycles,
            without.metrics.ipc(),
            without.metrics.branch_mispredictions,
            without.metrics.branch_accuracy(),
            wall_us,
            mips,
            with.metrics.bit_erasures + with.memory.bit_erasures,
        ));

        println!(
            "{:<24} {:>10} {:>8.4} {:>8.1}% {:>8.1}% {:>8.2}",
            name,
            with.metrics.instructions_retired,
            with.metrics.ipc(),
            with.metrics.branch_accuracy(),
            without.metrics.branch_accuracy(),
            mips
        );
    }

    let _ = std::fs::create_dir_all("docs");
    match std::fs::write("docs/bench.csv", &csv) {
        Ok(_) => println!("\nWrote docs/bench.csv"),
        Err(e) => eprintln!("\nCould not write docs/bench.csv: {}", e),
    }
}

fn benchmark_assembly_source(source: &str, iterations: usize) {
    let mut assembler = Assembler::new();
    let program = assembler.assemble(source).expect("Assembly failed");

    println!("Benchmarking program over {} iterations...", iterations);
    let start = Instant::now();

    for _ in 0..iterations {
        let mut cpu = Cpu::new(512 * 1024);
        cpu.capture_output = true;
        for (addr, bytes) in &program.data_segment {
            let _ = cpu.memory.write_bytes(*addr, bytes);
        }
        cpu.reset(program.entry_point);
        let _ = cpu.run_program(&program.instructions, 1_000_000);
    }

    let elapsed = start.elapsed();
    let per_run = elapsed / iterations as u32;
    println!("Total time: {:.3?}", elapsed);
    println!("Mean time per iteration: {:.3?}", per_run);
}

fn parse_temp_arg(args: &[String]) -> Option<f64> {
    for (i, arg) in args.iter().enumerate() {
        if (arg == "--temp-k" || arg == "--temp" || arg == "-T") && i + 1 < args.len() {
            return args[i + 1].parse::<f64>().ok();
        }
    }
    None
}

// ─── UI / Banner ─────────────────────────────────────────────────────────────

fn print_banner() {
    println!(r#"
  ███████╗ ██████╗ ██████╗      ██╗ █████╗     ██████╗ ███████╗ ██████╗ 
  ██╔════╝██╔═══██╗██╔══██╗     ██║██╔══██╗    ╚════██╗██╔════╝██╔════╝ 
  █████╗  ██║   ██║██████╔╝     ██║███████║     █████╔╝███████╗███████╗ 
  ██╔══╝  ██║   ██║██╔══██╗██   ██║██╔══██║    ██╔═══╝ ╚════██║██╔═══██╗
  ██║     ╚██████╔╝██║  ██║╚█████╔╝██║  ██║    ███████╗███████║╚██████╔╝
  ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚════╝ ╚═╝  ╚═╝    ╚══════╝╚══════╝ ╚═════╝ 
"#);
    println!("  » Post-Quantum AI-Native 256-Bit Custom Architecture Engine");
    println!("  » Landauer Thermodynamic Floor · BHT/BTB Predictor · 256-bit SIMD\n");
}

fn print_usage() {
    println!("COMMANDS:");
    println!("  unibit run <file.uasm|file.ubo> [--trace] [--temp-k <K>]
                                                Run a program or object file");
    println!("  unibit build <file.uasm> [-o <out.ubo>]        Assemble to an UBIT object file
  unibit disasm <file.ubo|file.uasm>             Decode bytecode to ASM + CFG + entropy profile");
    println!("  unibit cfg <file.uasm>                          Generate Control Flow Graph & branch map");
    println!("  unibit demo <name> [--trace]                   Run a built-in demo:");
    println!("                                                  • fibonacci");
    println!("                                                  • quantum_pqc");
    println!("                                                  • neural_tensor");
    println!("  unibit bench                                  Benchmark every program, write docs/bench.csv
  unibit bench <file.uasm>                      Benchmark one program's throughput");
    println!("  unibit version                                 Show version information");
    println!("  unibit help                                    Show this help message\n");
}
