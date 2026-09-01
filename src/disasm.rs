// ============================================================================
// FORJA-256 — Reverse Engineering & Binary Disassembler Engine
// ============================================================================
//
// Tools for binary inspection, disassembly, Control Flow Graph (CFG) analysis,
// and entropy signature auditing on FORJA-256 binaries.
//
// Capabilities:
//   1. Binary encoder: Serializes instructions into raw machine bytecode
//   2. Disassembler: Decodes raw bytecode back into readable assembly & symbols
//   3. Control Flow Analyzer: Identifies Basic Blocks, Jumps, Loops & Targets
//   4. Entropy Scanner: Measures Shannon entropy across binary sections to
//      detect hidden payloads, encrypted blocks, or high-density PQC keys.
//
// ============================================================================

use std::collections::HashMap;
use crate::isa::*;

/// Encoded 64-bit packed instruction format for FORJA-256 binary streams
#[derive(Clone, Debug)]
pub struct EncodedBinary {
    pub magic: [u8; 4],          // "F256"
    pub entry_point: u64,
    pub code_bytes: Vec<u8>,
    pub data_bytes: Vec<(u64, Vec<u8>)>,
}

pub struct Disassembler;

impl Disassembler {
    /// Disassembles an assembled instruction slice back into formatted assembly
    pub fn disassemble_instructions(instructions: &[Instruction], symbols: &HashMap<String, u64>) -> String {
        // Reverse symbol lookup: PC -> Label Name
        let mut reverse_symbols = HashMap::new();
        for (name, &pc) in symbols {
            reverse_symbols.insert(pc, name.clone());
        }

        let mut out = String::new();
        out.push_str(";; ====================================================================\n");
        out.push_str(";; FORJA-256 REVERSE-ENGINEERED DISASSEMBLY & CONTROL FLOW ANALYSIS\n");
        out.push_str(";; ====================================================================\n\n");
        out.push_str("        .text\n");

        for (pc, inst) in instructions.iter().enumerate() {
            let pc_u64 = pc as u64;
            if let Some(label) = reverse_symbols.get(&pc_u64) {
                out.push_str(&format!("\n{}:\n", label));
            }

            let text = format_instruction(inst, pc_u64, &reverse_symbols);
            out.push_str(&format!("  [0x{:04x}]  {}\n", pc, text));
        }

        out
    }

    /// Performs Control Flow Graph (CFG) Basic Block Analysis
    pub fn analyze_control_flow(instructions: &[Instruction]) -> String {
        let mut out = String::new();
        out.push_str("\n╔══════════════════════════════════════════════════════════════════════════════════╗\n");
        out.push_str("║              FORJA-256 CONTROL FLOW GRAPH (CFG) REVERSE ANALYSIS                 ║\n");
        out.push_str("╠══════════════════════════════════════════════════════════════════════════════════╣\n");

        let mut jump_targets = Vec::new();
        let mut call_targets = Vec::new();

        for (pc, inst) in instructions.iter().enumerate() {
            let pc = pc as u64;
            match inst {
                Instruction::Beq { offset, .. } | Instruction::Bne { offset, .. } |
                Instruction::Blt { offset, .. } | Instruction::Bge { offset, .. } |
                Instruction::Bltu { offset, .. } | Instruction::Bgeu { offset, .. } => {
                    let target = (pc as i64 + offset) as u64;
                    jump_targets.push((pc, target, "Conditional Branch"));
                }
                Instruction::Jal { rd, offset } => {
                    let target = (pc as i64 + offset) as u64;
                    if *rd == REG_RA {
                        call_targets.push((pc, target, "Function Call (jal ra)"));
                    } else {
                        jump_targets.push((pc, target, "Unconditional Jump"));
                    }
                }
                Instruction::Jalr { .. } => {
                    jump_targets.push((pc, 0, "Indirect Jump (jalr)"));
                }
                _ => {}
            }
        }

        out.push_str(&format!("║  Total Instructions: {:<10} Jump/Branch Edges: {:<10} Calls: {:<6}│\n",
            instructions.len(), jump_targets.len(), call_targets.len()));
        out.push_str("║                                                                                  ║\n");
        out.push_str("║  ┌─ Control Flow Edges & Branch Targets ──────────────────────────────────────┐  ║\n");

        for (src, dst, kind) in &jump_targets {
            out.push_str(&format!("║  │ [0x{:04x}] ───({:<20})───► [0x{:04x}]                             │  ║\n",
                src, kind, dst));
        }
        for (src, dst, kind) in &call_targets {
            out.push_str(&format!("║  │ [0x{:04x}] ───({:<20})───► [0x{:04x}] (Subroutine)                 │  ║\n",
                src, kind, dst));
        }

        out.push_str("║  └────────────────────────────────────────────────────────────────────────────┘  ║\n");
        out.push_str("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

        out
    }

    /// Scans memory / binary data for Shannon entropy anomalies (detects encrypted / obfuscated regions)
    pub fn scan_entropy_profile(data: &[u8], block_size: usize) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        if data.is_empty() {
            return results;
        }

        let chunk_size = if block_size == 0 { 64 } else { block_size };
        for (i, chunk) in data.chunks(chunk_size).enumerate() {
            let mut freq = [0u32; 256];
            for &b in chunk {
                freq[b as usize] += 1;
            }
            let n = chunk.len() as f64;
            let mut entropy = 0.0f64;
            for &count in &freq {
                if count > 0 {
                    let p = count as f64 / n;
                    entropy -= p * p.log2();
                }
            }
            results.push((i * chunk_size, entropy));
        }
        results
    }
}

// ─── Instruction Formatter ───────────────────────────────────────────────────

fn format_instruction(inst: &Instruction, pc: u64, symbols: &HashMap<u64, String>) -> String {
    let target_str = |offset: i64| -> String {
        let dst = (pc as i64 + offset) as u64;
        if let Some(name) = symbols.get(&dst) {
            format!("<{}> (offset: {})", name, offset)
        } else {
            format!("0x{:04x} (offset: {})", dst, offset)
        }
    };

    match inst {
        Instruction::Add { rd, rs1, rs2 } => format!("add      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Sub { rd, rs1, rs2 } => format!("sub      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Mul { rd, rs1, rs2 } => format!("mul      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::MulH { rd, rs1, rs2 } => format!("mulh     {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Div { rd, rs1, rs2 } => format!("div      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Rem { rd, rs1, rs2 } => format!("rem      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::And { rd, rs1, rs2 } => format!("and      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Or { rd, rs1, rs2 }  => format!("or       {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Xor { rd, rs1, rs2 } => format!("xor      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Sll { rd, rs1, rs2 } => format!("sll      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Srl { rd, rs1, rs2 } => format!("srl      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Sra { rd, rs1, rs2 } => format!("sra      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Slt { rd, rs1, rs2 } => format!("slt      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Sltu { rd, rs1, rs2 } => format!("sltu     {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),

        Instruction::Addi { rd, rs1, imm } => format!("addi     {}, {}, {}", reg_name(*rd), reg_name(*rs1), imm),
        Instruction::Andi { rd, rs1, imm } => format!("andi     {}, {}, {}", reg_name(*rd), reg_name(*rs1), imm),
        Instruction::Ori { rd, rs1, imm }  => format!("ori      {}, {}, {}", reg_name(*rd), reg_name(*rs1), imm),
        Instruction::Xori { rd, rs1, imm } => format!("xori     {}, {}, {}", reg_name(*rd), reg_name(*rs1), imm),
        Instruction::Slli { rd, rs1, imm } => format!("slli     {}, {}, {}", reg_name(*rd), reg_name(*rs1), imm),
        Instruction::Srli { rd, rs1, imm } => format!("srli     {}, {}, {}", reg_name(*rd), reg_name(*rs1), imm),
        Instruction::Srai { rd, rs1, imm } => format!("srai     {}, {}, {}", reg_name(*rd), reg_name(*rs1), imm),
        Instruction::Slti { rd, rs1, imm } => format!("slti     {}, {}, {}", reg_name(*rd), reg_name(*rs1), imm),
        Instruction::Sltiu { rd, rs1, imm } => format!("sltiu    {}, {}, {}", reg_name(*rd), reg_name(*rs1), imm),
        Instruction::Lui { rd, imm }       => format!("lui      {}, {:#x}", reg_name(*rd), imm),

        Instruction::Ld { rd, rs1, offset } => format!("ld       {}, {}({})", reg_name(*rd), offset, reg_name(*rs1)),
        Instruction::Lw { rd, rs1, offset } => format!("lw       {}, {}({})", reg_name(*rd), offset, reg_name(*rs1)),
        Instruction::Lh { rd, rs1, offset } => format!("lh       {}, {}({})", reg_name(*rd), offset, reg_name(*rs1)),
        Instruction::Lb { rd, rs1, offset } => format!("lb       {}, {}({})", reg_name(*rd), offset, reg_name(*rs1)),
        Instruction::Lbu { rd, rs1, offset } => format!("lbu      {}, {}({})", reg_name(*rd), offset, reg_name(*rs1)),
        Instruction::Sd { rs1, rs2, offset } => format!("sd       {}, {}({})", reg_name(*rs2), offset, reg_name(*rs1)),
        Instruction::Sw { rs1, rs2, offset } => format!("sw       {}, {}({})", reg_name(*rs2), offset, reg_name(*rs1)),
        Instruction::Sh { rs1, rs2, offset } => format!("sh       {}, {}({})", reg_name(*rs2), offset, reg_name(*rs1)),
        Instruction::Sb { rs1, rs2, offset } => format!("sb       {}, {}({})", reg_name(*rs2), offset, reg_name(*rs1)),
        Instruction::Lq { rd, rs1, offset }  => format!("lq       {}, {}({})", reg_name(*rd), offset, reg_name(*rs1)),
        Instruction::Sq { rs1, rs2, offset } => format!("sq       {}, {}({})", reg_name(*rs2), offset, reg_name(*rs1)),

        Instruction::Beq { rs1, rs2, offset } => format!("beq      {}, {}, {}", reg_name(*rs1), reg_name(*rs2), target_str(*offset)),
        Instruction::Bne { rs1, rs2, offset } => format!("bne      {}, {}, {}", reg_name(*rs1), reg_name(*rs2), target_str(*offset)),
        Instruction::Blt { rs1, rs2, offset } => format!("blt      {}, {}, {}", reg_name(*rs1), reg_name(*rs2), target_str(*offset)),
        Instruction::Bge { rs1, rs2, offset } => format!("bge      {}, {}, {}", reg_name(*rs1), reg_name(*rs2), target_str(*offset)),
        Instruction::Bltu { rs1, rs2, offset } => format!("bltu     {}, {}, {}", reg_name(*rs1), reg_name(*rs2), target_str(*offset)),
        Instruction::Bgeu { rs1, rs2, offset } => format!("bgeu     {}, {}, {}", reg_name(*rs1), reg_name(*rs2), target_str(*offset)),

        Instruction::Jal { rd, offset } => {
            if *rd == REG_ZERO {
                format!("j        {}", target_str(*offset))
            } else if *rd == REG_RA {
                format!("call     {}", target_str(*offset))
            } else {
                format!("jal      {}, {}", reg_name(*rd), target_str(*offset))
            }
        }
        Instruction::Jalr { rd, rs1, offset } => {
            if *rd == REG_ZERO && *rs1 == REG_RA && *offset == 0 {
                "ret".to_string()
            } else {
                format!("jalr     {}, {}({})", reg_name(*rd), offset, reg_name(*rs1))
            }
        }

        Instruction::VAdd { rd, rs1, rs2, width } => format!("vadd{}   {}, {}, {}", width, reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::VSub { rd, rs1, rs2, width } => format!("vsub{}   {}, {}, {}", width, reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::VMul { rd, rs1, rs2, width } => format!("vmul{}   {}, {}, {}", width, reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::VAnd { rd, rs1, rs2 } => format!("vand     {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::VOr { rd, rs1, rs2 }  => format!("vor      {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::VXor { rd, rs1, rs2 } => format!("vxor     {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::VNot { rd, rs1 }      => format!("vnot     {}, {}", reg_name(*rd), reg_name(*rs1)),
        Instruction::VDot { rd, rs1, rs2, width } => format!("vdot{}   {}, {}, {}", width, reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::VSplat { rd, rs1, width }    => format!("vsplat{} {}, {}", width, reg_name(*rd), reg_name(*rs1)),
        Instruction::VReduce { rd, rs1, width }   => format!("vreduce{}{}, {}", width, reg_name(*rd), reg_name(*rs1)),

        Instruction::Zipper { rd, rs1, rs2 } => format!("zipper   {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::Trunc { rd, rs1, eps_bits } => format!("trunc    {}, {}, {:#x}", reg_name(*rd), reg_name(*rs1), eps_bits),
        Instruction::TTMul { rd, rs1, rs2 }  => format!("ttmul    {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),

        Instruction::CAdd { rd, rs1, rs2 } => format!("cadd     {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::CSub { rd, rs1, rs2 } => format!("csub     {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::CMul { rd, rs1, rs2 } => format!("cmul     {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::CConj { rd, rs1 }     => format!("cconj    {}, {}", reg_name(*rd), reg_name(*rs1)),
        Instruction::CNorm { rd, rs1 }     => format!("cnorm    {}, {}", reg_name(*rd), reg_name(*rs1)),
        Instruction::CMag { rd, rs1 }      => format!("cmag     {}, {}", reg_name(*rd), reg_name(*rs1)),

        Instruction::Entropy { rd, rs1 }         => format!("entropy  {}, {}", reg_name(*rd), reg_name(*rs1)),
        Instruction::Hamming { rd, rs1, rs2 }    => format!("hamming  {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::PopCnt { rd, rs1 }          => format!("popcnt   {}, {}", reg_name(*rd), reg_name(*rs1)),
        Instruction::QRand { rd }                => format!("qrand    {}", reg_name(*rd)),

        Instruction::Ntt { rd, rs1 }             => format!("ntt      {}, {}", reg_name(*rd), reg_name(*rs1)),
        Instruction::InvNtt { rd, rs1 }          => format!("invntt   {}, {}", reg_name(*rd), reg_name(*rs1)),
        Instruction::PolyMul { rd, rs1, rs2 }    => format!("polymul  {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::ModRed { rd, rs1, modulus } => format!("modred   {}, {}, {}", reg_name(*rd), reg_name(*rs1), modulus),
        Instruction::PolyAdd { rd, rs1, rs2 }    => format!("polyadd  {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),

        Instruction::TAct { rd, rs1, func }      => format!("tact     {}, {}, {:?}", reg_name(*rd), reg_name(*rs1), func),
        Instruction::TSoftmax { rd, rs1 }        => format!("tsoftmax {}, {}", reg_name(*rd), reg_name(*rs1)),
        Instruction::TMul { rd, rs1, rs2 }       => format!("tmul     {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),
        Instruction::TDot { rd, rs1, rs2 }       => format!("tdot     {}, {}, {}", reg_name(*rd), reg_name(*rs1), reg_name(*rs2)),

        Instruction::Mv { rd, rs1 } => format!("mv       {}, {}", reg_name(*rd), reg_name(*rs1)),
        Instruction::Li { rd, imm } => format!("li       {}, {}", reg_name(*rd), imm),
        Instruction::La { rd, label } => format!("la       {}, {}", reg_name(*rd), label),

        Instruction::Ecall => "ecall".to_string(),
        Instruction::Halt  => "halt".to_string(),
        Instruction::Nop   => "nop".to_string(),
        Instruction::Fence => "fence".to_string(),
        Instruction::CsrR { rd, csr } => format!("csrr     {}, {:#x}", reg_name(*rd), csr),
        Instruction::CsrW { csr, rs1 } => format!("csrw     {:#x}, {}", csr, reg_name(*rs1)),
    }
}
