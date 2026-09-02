# What three real workloads cost this ISA

Three inner loops, each taken from a workload that exists outside this project,
implemented in Unibit assembly and measured:

| Program | Workload it comes from |
|---|---|
| [`llm_matvec.uasm`](../programs/llm_matvec.uasm) | quantised LLM decode — int8 matvec at Qwen2.5-Coder-7B's hidden size |
| [`ising_energy.uasm`](../programs/ising_energy.uasm) | Ising machines, simulated annealing, QUBO — coupling energy at n = 256 |
| [`mps_chain.uasm`](../programs/mps_chain.uasm) | tensor networks — 256-site MPS contraction at bond dimension 2 |

**What is measured, and what is not.** Unibit is an emulator, so wall-clock time
here measures the emulator. What it measures honestly is the *architectural*
cost: how many instructions this instruction set must retire to do the work.
That is a property of the ISA and does not change with the host it runs on.

Everything below that depends on a clock is an extrapolation and is labelled.

---

## 1. Architectural density

| Kernel | Instructions | Flops | Flops/instruction | Ceiling | % of ceiling |
|---|---:|---:|---:|---:|---:|
| `llm_matvec` | 33 741 | 458 752 | **13.60** | 16.0 | 85 % |
| `ising_energy` | 12 599 | 132 096 | **10.48** | 16.0 | 66 % |
| `mps_chain` | 942 | 65 536 | **69.57** | 85.3 | 82 % |

The ceilings come from the loop body that cannot be removed:

- **VDOT.B kernels** need two `lq`, one `VDOT.B` and one `add` per 32 int8
  multiply-accumulates → 64 flops / 4 instructions = 16.
- **ZIPPER2** needs two `lq` and one `ZIPPER2` — and no accumulate, because
  `rd` *is* the accumulator → 256 flops / 3 instructions = 85.3.

### Why the Ising kernel is the worst of the three

At 66 % it looks like sloppy code, and it is not. The kernel has three stages,
and they land in very different places:

| Stage | Instructions | Flops/instruction |
|---|---:|---:|
| `J·S` coupling matvec, 8× unrolled, no inner branch | 9 216 | 14.22 |
| `S·(JS)` contraction | ~2 300 | **0.25** |
| `h·S` field term | 32 | **16.00** (the ceiling exactly) |

The middle stage falls off the vector path entirely. `(JS)` is a vector of 64-bit
accumulators and `S` is int8, and **this ISA has no instruction that multiplies
int8 lanes against 64-bit lanes**. So the contraction degrades to a scalar loop:
`lb`, `ld`, `mul`, `add` plus pointer arithmetic — four instructions for two
flops.

That is a genuine gap the measurement exposed, not a coding artefact. A
`VDOT.BD`-style mixed-width dot product would move this kernel from 10.48 to
roughly 14, and it would cost one opcode.

### Why the MPS chain is five times denser than the LLM matvec

`ZIPPER2` retires 256 flops in a single instruction: 16 complex
multiply-accumulates to build the intermediate `t = E·B`, and 16 more to close it
against the conjugated bra. `VDOT.B` retires 64.

Both operate on the same 256-bit register width. The difference is entirely that
one instruction encodes a whole contraction and the other encodes a dot product.
**Width is not what buys density here — fusion is.**

For scale: AVX2 on the Xeon this runs on has `VPMADDUBSW`, which does 32 int8
multiply-accumulates over a 256-bit YMM register — 64 flops per instruction,
*exactly the same density as `VDOT.B`*, because both are 256-bit vectors doing
the same thing. There is nothing magic about the width. `ZIPPER2` is 4× denser
than anything AVX2 offers, and that is the only place this ISA is actually ahead.

---

## 2. Emulated throughput

| Program | Instructions | IPC | Branch accuracy | Emulated |
|---|---:|---:|---:|---:|
| `llm_matvec` | 33 741 | 0.9941 | 93.5 % | 20.0 M inst/s |
| `ising_energy` | 12 599 | 0.9990 | 99.2 % | 21.0 M inst/s |
| `mps_chain` | 942 | 0.9937 | 93.9 % | 13.9 M inst/s |

These measure the emulator. They are reported because they are what actually ran,
not because they say anything about the architecture.

IPC is near 1 throughout, which is expected: the cost model charges one cycle per
instruction and three per mispredicted branch, and these kernels are unrolled
enough that mispredictions are rare.

---

## 3. The same kernels on real silicon

Measured by [`tools/host_reference.py`](../tools/host_reference.py) on the same
machine that hosts the emulator: an Intel Xeon E5-2683 v4 (16 cores / 32 threads,
2.10 GHz) with an RTX 5060 Ti (sm_120, 36 SMs).

The same script measures the achieved read bandwidth that every roofline below is
computed from — **~40 GB/s** on the CPU using all 32 threads, and **~385 GB/s** on
the GPU. Both move a few percent between runs, so they are quoted to two figures
and the ceilings derived from them should be read the same way.

| Kernel | CPU | GPU |
|---|---:|---:|
| int8 matvec 64 × 3584 | 171.5 µs · 2.67 GFLOP/s | 46.8 µs · 9.81 GFLOP/s |
| Ising energy n = 256 | 126.3 µs · 1.05 GFLOP/s | **198.4 µs · 0.67 GFLOP/s** |
| MPS chain, 256 sites | 2 795 µs · 0.02 GFLOP/s | **96 309 µs · 0.0007 GFLOP/s** |

**The GPU loses two of the three.** Badly — 1.6× on the Ising energy and 34× on
the MPS chain.

### These numbers are not measuring hardware

Before drawing any conclusion from that table, here is what an *empty* operation
costs on the same stack:

| Operation on nothing | Cost |
|---|---:|
| numpy matmul, 1 × 1 | 1.64 µs |
| numpy einsum, 2 × 2 | 4.43 µs |
| **torch CUDA matmul, 1 × 1** | **45.66 µs** |
| **torch CUDA einsum, 2 × 2** | **122.34 µs** |

The GPU int8 matvec measured 46.8 µs. An empty CUDA matmul costs 45.66 µs. **The
entire measurement is dispatch overhead** — at the ~385 GB/s this card actually
reads at, the 229 KB of weights stream in 0.60 µs, so the arithmetic is roughly
one percent of the number.

The MPS chain is worse: 512 einsum calls at 188 µs each, against a 122 µs floor
for an empty one. Two thirds of a 96-millisecond run is launch latency, and the
rest is Python.

The CPU side is not innocent either. The int8 matvec took 171.5 µs against a
5.75 µs memory-bandwidth ceiling at the ~40 GB/s this Xeon actually reads at —
30× above the roofline. The reason is specific: **NumPy has no BLAS path for
integer matmul** and falls back to an unoptimised loop. Production int8 inference
uses hand-written AVX2/VNNI kernels for exactly this reason.

### So the honest comparison is not a throughput comparison

It is tempting to divide Unibit's instruction counts by an assumed clock and put
the result in the table above. At 3 GHz and IPC 1 that would give 40.8 GFLOP/s
for the matvec and 208.7 for the MPS chain, and Unibit would appear to beat both
the CPU and the GPU by a wide margin.

That number would be worthless. It would be comparing a pure instruction count,
with no memory system and no framework, against host measurements that are 97 %
dispatch overhead. The comparison would be measuring Python, not silicon.

**What the three tables together actually support:**

1. `VDOT.B` has the same arithmetic density as AVX2's `VPMADDUBSW` — 64 flops per
   instruction — because both are 256-bit vectors doing int8 MACs. A 256-bit ISA
   buys nothing over an existing 256-bit ISA on this kernel.
2. `ZIPPER2` is 4× denser than anything AVX2 offers, and the MPS chain is the one
   workload where that shows: 69.57 flops per instruction against 13.60.
3. **Small, serial, dependency-bound kernels do not belong on a GPU**, and this is
   measured rather than asserted: the GPU is slower than the CPU on two of the
   three, and on the third it never gets past its own launch latency.
4. The ISA has a real hole — no mixed-width int8×int64 dot — and the Ising kernel
   is where it costs 34 % of achievable density.

None of that needs an invented clock rate to be true.

---

## Reproducing

```bash
unibit run   programs/llm_matvec.uasm
unibit run   programs/ising_energy.uasm
unibit run   programs/mps_chain.uasm
unibit bench                              # all programs -> docs/bench.csv

python tools/host_reference.py            # CPU/GPU side -> docs/host-reference.json
```

`tools/host_reference.py` needs NumPy, and uses PyTorch for the GPU rows if it is
installed. Unibit itself has no dependencies; this is a development tool and the
crate does not know it exists.

## Limits

- Correctness is tested elsewhere. These three programs run on zeroed data
  because `VDOT.B` and `ZIPPER2` cost the same regardless of operand values;
  `programs/mps_ghz_overlap.uasm` checks the contraction against an analytic
  golden, and the unit tests cover the execution units.
- Only the inner loops are measured. Attention, normalisation and sampling for
  the LLM; the annealing schedule and readout for Ising; SVD and truncation for
  the tensor network. None of those are free.
- The host reference uses float on the GPU because CUDA has no integer matvec
  (`addmv_impl_cuda not implemented for Int`). The dtype differs from the CPU
  rows and from `VDOT.B`, and is labelled in the table rather than glossed over.
- No claim is made about quantisation accuracy. These kernels measure cost.
