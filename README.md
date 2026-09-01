# FORJA-256 — Post-Quantum AI-Native 256-Bit Architecture

> *"No programas el metal. CREAS el metal."*

**FORJA-256** es una arquitectura y procesador de 256 bits diseñado desde cero en Rust bare-metal (zero-dependencies), concebido específicamente para la intersección de **Post-Quantum Cryptography (PQC)**, **Aceleración de Inteligencia Artificial** y **Termodinámica de la Información (Principio de Landauer)**.

---

## ⚡ ¿Qué hace a FORJA-256 Diferente?

A diferencia de las arquitecturas RISC convencionales (x86, ARM, RISC-V), FORJA-256 integra directamente en la ISA:

1. **Registros de 256-bit con Modos Semánticos:**
   - `Scalar`: 1 × 64-bit (compatibilidad RISC clásica)
   - `Vector .d`: 4 × 64-bit (f64 / enteror)
   - `Vector .w`: 8 × 32-bit (fp32 / int32)
   - `Vector .h`: 16 × 16-bit (fp16 / bf16 AI weights)
   - `Vector .b`: 32 × 8-bit (int8 quantized AI inference)
   - `Complex`: 2 × 128-bit pares de números complejos para simulación de estados cuánticos.
   - `Poly`: 4 × 64-bit coeficientes de anillos polinomiales $\mathcal{R}_q = \mathbb{Z}_q[X]/(X^4+1)$.

2. **Instrucciones de Teoría de la Información Nativas:**
   - `ENTROPY rd, rs1`: Calcula en hardware la entropía de Shannon ($H(X) = -\sum p \log_2 p$) del contenido del registro.
   - `HAMMING rd, rs1, rs2`: Distancia de Hamming a nivel de 256 bits simultáneos.
   - `POPCNT rd, rs1`: Conteo instantáneo de bits activos (population count).
   - `QRAND rd`: Generador de pseudo-entropía de alta aleatoriedad para estados de simulación.

3. **Acelerador de Criptografía Post-Cuántica (Lattice PQC):**
   - `NTT rd, rs1` / `INVNITT rd, rs1`: Number Theoretic Transform directa en hardware.
   - `POLYMUL rd, rs1, rs2`: Multiplicación polinomial negacíclica modular sobre anillos $\mathbb{Z}_q$ ($q=3329$ Kyber / $q=8380417$ Dilithium).
   - `POLYADD rd, rs1, rs2`: Suma modular de vectores polinomiales.

4. **Acelerador Tensor & Activaciones de Redes Neuronales:**
   - `TACT rd, rs1, <func>`: Activaciones no lineales vectorizadas (`relu`, `gelu`, `silu`, `sigmoid`, `tanh`).
   - `TSOFTMAX rd, rs1`: Distribución de probabilidad Softmax calculada sobre 4 carriles vectoriales.
   - `VDOT.D / VDOT.W`: Productos punto vectoriales en un solo ciclo.

5. **Motor de Termodinámica de Landauer:**
   - Rastreo en tiempo real de cada bit destruido/reescrito de forma irreversible.
   - Cálculo del trabajo termodinámico disipado en zeptojulios ($E \ge k_B T \ln 2$) a temperatura configurable $T$ (300K ambiente, 4K helio líquido, 15mK dilución cuántica).

---

## 🛠️ Estructura del Proyecto

```
FORJA/
├── Cargo.toml                  # Zero dependencies · Pure Rust
├── README.md                   # Esta documentación
├── programs/                   # Programas en ensamblador FORJA (.fja)
│   ├── fibonacci.fja           # Benchmark Fibonacci
│   ├── quantum_pqc.fja         # Demostración QRAND + Entropía + PQC
│   └── mandelbrot.fja          # Render ASCII de fractal Mandelbrot
└── src/
    ├── lib.rs                  # Exportaciones públicas de la biblioteca
    ├── main.rs                 # CLI Runner, Demos embebidos y Benchmark
    ├── isa.rs                  # Definición de la ISA de 256 bits y registros
    ├── alu.rs                  # Unidades de ejecución: Scalar, SIMD, Complex, PQC, Tensor
    ├── memory.rs               # Memoria little-endian configurable con métricas de bus
    ├── cpu.rs                  # Pipeline, predictor BHT/BTB, Landauer Tracker y Syscalls
    └── assembler.rs            # Ensamblador de dos pasadas, linker de etiquetas y directivas
```

---

## 🚀 Uso del CLI

### 1. Ejecutar Demos Nativos Embebidos

```bash
# Demo de Fibonacci y cálculo de trabajo de Landauer
cargo run -- demo fibonacci

# Demo Post-Quantum, Entropía de Shannon y Álgebra Compleja
cargo run -- demo quantum_pqc

# Demo SIMD de 256-bit y Activaciones Neuronales (GeLU / Softmax)
cargo run -- demo neural_tensor
```

### 2. Ejecutar un Archivo de Ensamblador

```bash
# Renderizar Mandelbrot en ASCII con el procesador
cargo run -- run programs/mandelbrot.fja

# Ejecutar con modo Trace (ciclo por ciclo)
cargo run -- run programs/fibonacci.fja --trace

# Configurar temperatura de operación (ej: 15 mK en refrigerador de dilución cuántica)
cargo run -- run programs/quantum_pqc.fja --temp-k 0.015
```

### 3. Benchmark de Rendimiento

```bash
cargo run -- bench programs/mandelbrot.fja
```

---

## 📊 Ejemplo de Reporte Microarquitectónico

Al finalizar cualquier ejecución, FORJA-256 genera un informe completo del estado físico de la máquina:

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      FORJA-256 PROCESSOR EXECUTION REPORT                        ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  ┌─ Microarchitecture & Pipeline ─────────────────────────────────────────────┐  ║
║  │ Total Instructions Retired:  265805         Cycles: 372926              │  ║
║  │ IPC (Instructions / Cycle):  0.7128         Stalls: 100317              │  ║
║  │ Branch Predictor (BHT/BTB):  34595  branches (3.34 % accuracy, 33439 miss)    │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                  ║
║  ┌─ Post-Quantum & Domain-Specific Units ─────────────────────────────────────┐  ║
║  │ Vector SIMD (256-bit):       0              Complex (Quantum): 0             │  ║
║  │ Lattice PQC (NTT/Poly):      0              Neural Tensor:     0             │  ║
║  │ Information Theory (Entropy):0              Memory Reads/Writes:0    /0     │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                  ║
║  ┌─ Landauer Thermodynamic Dissipation (Physics Engine) ──────────────────────┐  ║
║  │ Operating Temperature:       300.00  K (Ambient / Dilution Ref configurable) │  ║
║  │ Irreversible Bit Erasures:   3200766        bits flipped/erased           │  ║
║  │ Landauer Energy Floor (1b):  2.87098e-21    Joules (k_B·T·ln2)                  │  ║
║  │ Min Thermodynamic Work:      9.18933e-15    Joules (9189331.60 zeptoJoules)     │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 📜 Formato de Ensamblador (.fja)

```asm
        .data
msg:    .asciiz "Hello from custom 256-bit hardware!\n"

        .text
        .global _start

_start:
        ; Imprimir texto mediante syscall 3
        la    a0, msg
        li    a1, 38
        li    a7, 3
        ecall

        ; Instrucciones nativas PQC y de teoría de la información
        qrand   t0             ; Estado de pseudo-entropía cuántica
        entropy t1, t0         ; Entropía de Shannon de t0 en bits
        polymul t2, t0, t1     ; Multiplicación polinomial en anillo Kyber mod 3329

        halt
```

---

*Desarrollado con mentalidad bare-metal pura. Diseñado para traspasar los límites de la computación clásica.* ⚡🧠⚛️
