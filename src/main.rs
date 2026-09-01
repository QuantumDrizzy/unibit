// ============================================================================
// FORJA-256 — CLI & Processor Simulator Runtime
// ============================================================================
//
// Usage:
//   forja run <file.fja> [--trace] [--temp-k <K>]
//   forja demo [fibonacci | quantum_pqc | neural_tensor]
//   forja bench <file.fja>
//
// ============================================================================

use std::env;
use std::fs;
use std::process;
use std::time::Instant;

use forja::assembler::Assembler;
use forja::cpu::Cpu;

// ─── Embedded Demo Programs ──────────────────────────────────────────────────

const DEMO_FIBONACCI: &str = r#"
; ============================================================================
; DEMO: Fibonacci Sequence (FORJA-256)
; Computes first 25 Fibonacci numbers, tracking cycles and Landauer work.
; ============================================================================

        .data
msg_title: .asciiz "=== FORJA-256: Fibonacci Computation ===\n"
msg_comma: .asciiz ", "
msg_nl:    .asciiz "\n"

        .text
        .global _start

_start:
        ; Print title
        la    a0, msg_title
        li    a1, 41
        li    a7, 3
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
        li    a1, 2
        li    a7, 3
        ecall

        mv    a0, s2
        li    a7, 1          ; print_int
        ecall                ; prints 1
        la    a0, msg_comma
        li    a1, 2
        li    a7, 3
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
        li    a1, 2
        li    a7, 3          ; print_str
        ecall

        addi  t0, t0, -1     ; counter--
        j     fib_loop

done:
        la    a0, msg_nl
        li    a1, 1
        li    a7, 3
        ecall
        halt
"#;

const DEMO_QUANTUM_PQC: &str = r#"
; ============================================================================
; DEMO: Quantum State & Post-Quantum Lattice PQC (FORJA-256)
; Demonstrates:
;   1. QRAND + ENTROPY instructions (Shannon entropy of 256-bit state)
;   2. Complex arithmetic: (3 + 4i) * (1 - 2i) = 11 - 2i
;   3. Lattice PQC: Negacyclic polynomial multiplication mod Q=3329
; ============================================================================

        .data
msg_banner:   .asciiz "=== FORJA-256: Quantum & PQC Accelerator Demo ===\n"
msg_ent:      .asciiz "[1] Quantum-Random 256-bit State Shannon Entropy: "
msg_bits:     .asciiz " bits\n"
msg_cmplx:    .asciiz "[2] Quantum Complex Conjugate Result:\n    "
msg_pqc:      .asciiz "[3] Post-Quantum Lattice Polynomial Multiplication mod 3329:\n    "
msg_done:     .asciiz "[OK] All Post-Quantum Instructions Executed Successfully!\n"
msg_nl:       .asciiz "\n"

        .text
        .global _start

_start:
        la    a0, msg_banner
        li    a1, 50
        li    a7, 3
        ecall

        ; ── 1. Information Theory Engine ──
        qrand   t0                 ; Fill t0 with quantum-random pseudo-entropy
        entropy t1, t0             ; t1 = Shannon entropy in bits (f64 bits)

        la    a0, msg_ent
        li    a1, 50
        li    a7, 3
        ecall

        mv    a0, t1
        li    a7, 5                ; print_f64
        ecall

        la    a0, msg_bits
        li    a1, 6
        li    a7, 3
        ecall

        ; ── 2. Complex Arithmetic (Quantum Unit) ──
        cmul  t4, t0, t1           ; t4 = complex multiplication
        cconj t5, t4               ; t5 = complex conjugate

        la    a0, msg_cmplx
        li    a1, 42
        li    a7, 3
        ecall

        li    a0, 5                ; print t5 (register 5)
        li    a7, 20               ; print_reg256
        ecall

        la    a0, msg_nl
        li    a1, 1
        li    a7, 3
        ecall

        ; ── 3. Post-Quantum Lattice Cryptography ──
        ; Polynomial multiplication in ring R_q (q=3329)
        polymul t6, t0, t1
        ntt     a2, t6
        invntt  a3, a2

        la    a0, msg_pqc
        li    a1, 66
        li    a7, 3
        ecall

        li    a0, 28               ; print t6 (reg 28)
        li    a7, 20
        ecall

        la    a0, msg_nl
        li    a1, 1
        li    a7, 3
        ecall

        la    a0, msg_done
        li    a1, 58
        li    a7, 3
        ecall

        halt
"#;

const DEMO_NEURAL_TENSOR: &str = r#"
; ============================================================================
; DEMO: 256-bit SIMD Vector & Neural Tensor Unit (FORJA-256)
; Demonstrates:
;   1. 32-way 8-bit vector arithmetic (int8 quantized AI inference)
;   2. Non-linear activation functions (GeLU, Softmax)
;   3. Vector sum-reduction and dot-products across 256 bits
; ============================================================================

        .data
msg_title:    .asciiz "=== FORJA-256: 256-bit SIMD & Neural Tensor Unit ===\n"
msg_vec:      .asciiz "[1] 32-way 8-bit Vector SIMD Addition (Int8 Quantized AI)\n"
msg_dot:      .asciiz "[2] 256-bit Vector Dot Product (vdot.d): "
msg_act:      .asciiz "[3] Neural Activation (GeLU + Softmax) Computed on 4 lanes\n"
msg_nl:       .asciiz "\n"

        .text
        .global _start

_start:
        la    a0, msg_title
        li    a1, 53
        li    a7, 3
        ecall

        ; ── 1. Vector SIMD ──
        vadd.b  t0, t0, t0         ; 32 parallel 8-bit adds
        vmul.w  t1, t0, t0         ; 8 parallel 32-bit multiplies
        vdot.d  t2, t1, t1         ; Dot product of 4 64-bit lanes

        la    a0, msg_vec
        li    a1, 58
        li    a7, 3
        ecall

        la    a0, msg_dot
        li    a1, 41
        li    a7, 3
        ecall

        mv    a0, t2
        li    a7, 1
        ecall

        la    a0, msg_nl
        li    a1, 1
        li    a7, 3
        ecall

        ; ── 2. Neural Activations ──
        tact    t3, t1, gelu       ; Apply GeLU non-linearity
        tsoftmax t4, t3            ; Compute Softmax probability distribution

        la    a0, msg_act
        li    a1, 59
        li    a7, 3
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
                eprintln!("Error: Missing file path. Usage: forja run <file.fja> [--trace] [--temp-k <K>]");
                process::exit(1);
            }
            let file_path = &args[2];
            let trace = args.iter().any(|a| a == "--trace" || a == "-t");
            let temp_k = parse_temp_arg(&args).unwrap_or(300.0);

            let source = match fs::read_to_string(file_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error reading file '{}': {}", file_path, e);
                    process::exit(1);
                }
            };

            run_assembly_source(&source, trace, temp_k);
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
                eprintln!("Error: Missing file path. Usage: forja disasm <file.fja>");
                process::exit(1);
            }
            let file_path = &args[2];
            let source = fs::read_to_string(file_path).unwrap_or_else(|e| {
                eprintln!("Error reading file '{}': {}", file_path, e);
                process::exit(1);
            });
            let mut assembler = Assembler::new();
            let program = assembler.assemble(&source).unwrap_or_else(|e| {
                eprintln!("Assembler error: {}", e);
                process::exit(1);
            });

            let disasm_output = forja::disasm::Disassembler::disassemble_instructions(&program.instructions, &program.symbols);
            println!("{}", disasm_output);

            let cfg_output = forja::disasm::Disassembler::analyze_control_flow(&program.instructions);
            println!("{}", cfg_output);
        }
        "cfg" => {
            if args.len() < 3 {
                eprintln!("Error: Missing file path. Usage: forja cfg <file.fja>");
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
            let cfg_output = forja::disasm::Disassembler::analyze_control_flow(&program.instructions);
            println!("{}", cfg_output);
        }
        "bench" => {
            if args.len() < 3 {
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
            println!("FORJA-256 Core Processor Simulator v0.1.0 (Post-Quantum AI-Native Architecture)");
        }
        "help" | "-h" | "--help" => {
            print_banner();
            print_usage();
        }
        other => {
            eprintln!("Unknown command: '{}'. Type 'forja help' for commands.", other);
            process::exit(1);
        }
    }
}

// ─── Execution Runner ────────────────────────────────────────────────────────

fn run_assembly_source(source: &str, trace: bool, temp_k: f64) {
    let mut assembler = Assembler::new();
    let program = match assembler.assemble(source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("\n[Assembler Error] {}\n", e);
            process::exit(1);
        }
    };

    let mut cpu = Cpu::new(1024 * 1024); // 1 MiB memory
    cpu.trace = trace;
    cpu.temp_k = temp_k;

    // Load data segment into RAM
    for (addr, bytes) in &program.data_segment {
        if let Err(e) = cpu.memory.write_bytes(*addr, bytes) {
            eprintln!("[Memory Error] Failed loading data at 0x{:x}: {}", addr, e);
            process::exit(1);
        }
    }

    cpu.reset(program.entry_point);

    let start = Instant::now();
    if let Err(e) = cpu.run_program(&program.instructions, 10_000_000) {
        eprintln!("\n[Runtime Trap/Error] {}\n", e);
    }
    let duration = start.elapsed();

    // Print final microarchitecture & Landauer report
    cpu.print_report();
    println!("Execution wall time: {:.3?}", duration);
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
    println!("  forja run <file.fja> [--trace] [--temp-k <K>]  Run an assembly program");
    println!("  forja disasm <file.fja>                       Reverse-engineer & decompile bytecode to ASM + CFG");
    println!("  forja cfg <file.fja>                          Generate Control Flow Graph & branch map");
    println!("  forja demo <name> [--trace]                   Run a built-in demo:");
    println!("                                                  • fibonacci");
    println!("                                                  • quantum_pqc");
    println!("                                                  • neural_tensor");
    println!("  forja bench <file.fja>                        Benchmark execution throughput");
    println!("  forja version                                 Show version information");
    println!("  forja help                                    Show this help message\n");
}
