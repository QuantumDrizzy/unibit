"""Host reference numbers for the kernels Unibit implements in assembly.

Unibit is an emulator, so its wall-clock time measures the emulator. To say
anything about the *architecture* the instruction counts have to be placed next
to what real silicon does with the same work. This script measures that side.

It is a development tool, not part of the crate: Unibit itself has no
dependencies. If torch is not installed the GPU section is skipped and the CPU
section still runs on NumPy.

    python tools/host_reference.py
"""
import json
import os
import platform
import time

import numpy as np

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = HAS_CUDA = False


def timed(fn, reps, sync=None):
    for _ in range(3):
        fn()
    if sync:
        sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if sync:
        sync()
    return (time.perf_counter() - t0) / reps


def main():
    if HAS_TORCH:
        torch.set_num_threads(os.cpu_count())
    sync = torch.cuda.synchronize if HAS_CUDA else None
    out = {"host": {}, "kernels": []}

    out["host"]["cpu"] = platform.processor()
    out["host"]["logical_cores"] = os.cpu_count()
    if HAS_CUDA:
        p = torch.cuda.get_device_properties(0)
        out["host"]["gpu"] = f"{p.name} sm_{p.major}{p.minor} {p.multi_processor_count} SMs"

    print("=" * 76)
    print("HOST REFERENCE — the same kernels on real silicon")
    print("=" * 76)
    print(f"  CPU  {out['host']['cpu']}  ({out['host']['logical_cores']} threads)")
    if HAS_CUDA:
        print(f"  GPU  {out['host']['gpu']}")
    print()

    def record(kernel, device, seconds, flops):
        row = {"kernel": kernel, "device": device,
               "seconds": seconds, "gflops": flops / seconds / 1e9}
        out["kernels"].append(row)
        print(f"  {kernel:<26} {device:<12} {seconds*1e6:10.1f} us   "
              f"{row['gflops']:9.2f} GFLOP/s")
        return row

    # ── 0. Achieved read bandwidth ───────────────────────────────────────
    # The roofline ceilings quoted in docs/kernel-cost.md are computed from
    # these two numbers, so they are measured here rather than cited from
    # somewhere else.
    print("  achieved read bandwidth")
    print("  " + "-" * 62)
    n = (1024 ** 3) // 4
    # torch, not numpy: numpy's sum is single-threaded and reports about a tenth
    # of what this Xeon actually reads, which would make every roofline below it
    # look flattering for the wrong reason.
    if HAS_TORCH:
        v = torch.ones(n, dtype=torch.float32)
        dt = timed(lambda: v.sum(), 6)
        bw_cpu = v.numel() * 4 / dt / 1e9
        label = f"torch x{torch.get_num_threads()}"
    else:
        v = np.ones(n, dtype=np.float32)
        dt = timed(lambda: v.sum(), 6)
        bw_cpu = v.nbytes / dt / 1e9
        label = "numpy x1"
    out["host"]["cpu_read_gbs"] = bw_cpu
    print(f"  {'CPU read (sum)':<26} {label:<12} "
          f"{dt*1e6:10.1f} us   {bw_cpu:9.1f} GB/s")

    if HAS_CUDA:
        g = torch.ones(n, dtype=torch.float32, device="cuda")
        dt = timed(lambda: g.sum(), 30, sync)
        bw_gpu = g.numel() * 4 / dt / 1e9
        out["host"]["gpu_read_gbs"] = bw_gpu
        print(f"  {'GPU read (sum)':<26} {'torch':<12} "
              f"{dt*1e6:10.1f} us   {bw_gpu:9.1f} GB/s")
        del g
        torch.cuda.empty_cache()
    del v
    print()

    # ── 1. int8 matvec, 64 x 3584 (programs/llm_matvec.uasm) ─────────────
    print("  int8 matvec, 64 x 3584  —  same shape as llm_matvec.uasm")
    print("  " + "-" * 62)
    rows, cols = 64, 3584
    flops = 2 * rows * cols

    w8 = np.random.randint(-8, 8, (rows, cols), dtype=np.int8)
    x8 = np.random.randint(-8, 8, cols, dtype=np.int8)
    # int32 accumulation: the honest equivalent of what VDOT.B does.
    wi, xi = w8.astype(np.int32), x8.astype(np.int32)
    record("int8 matvec", "CPU int32", timed(lambda: wi @ xi, 200), flops)

    if HAS_CUDA:
        # CUDA has no integer matvec ("addmv_impl_cuda not implemented for
        # Int"), so the GPU rows are float. The dtype differs from the CPU and
        # from VDOT.B and is labelled accordingly rather than quietly compared.
        wh = torch.from_numpy(wi).cuda().to(torch.float16)
        xh = torch.from_numpy(xi).cuda().to(torch.float16)
        record("int8 matvec", "GPU fp16",
               timed(lambda: wh @ xh, 300, sync), flops)
        wf, xf = wh.float(), xh.float()
        record("int8 matvec", "GPU fp32",
               timed(lambda: wf @ xf, 300, sync), flops)
        del wh, xh, wf, xf
        torch.cuda.empty_cache()

    # ── 2. Ising energy, n = 256 (programs/ising_energy.uasm) ────────────
    print("\n  Ising energy, n = 256  —  same shape as ising_energy.uasm")
    print("  " + "-" * 62)
    n = 256
    flops = 2 * n * n + 4 * n
    j8 = np.random.randint(-8, 8, (n, n), dtype=np.int8).astype(np.int32)
    s8 = np.random.choice(np.array([-1, 1], dtype=np.int32), n)
    h8 = np.random.randint(-4, 4, n).astype(np.int32)

    def ising_cpu():
        t = j8 @ s8
        return -0.5 * float(s8 @ t) - float(h8 @ s8)

    record("Ising energy n=256", "CPU int32", timed(ising_cpu, 200), flops)

    if HAS_CUDA:
        # float on the GPU for the same reason as above. Note this is exactly
        # the trade the Ising kernel avoids in integer: on a GPU the coupling
        # product runs in float and its error has to be accounted for.
        jg = torch.from_numpy(j8).cuda().float()
        sg = torch.from_numpy(s8).cuda().float()
        hg = torch.from_numpy(h8).cuda().float()

        def ising_gpu():
            t = jg @ sg
            return -0.5 * (sg @ t) - (hg @ sg)

        record("Ising energy n=256", "GPU fp32",
               timed(ising_gpu, 300, sync), flops)
        del jg, sg, hg
        torch.cuda.empty_cache()

    # ── 3. MPS chain, 256 sites at bond dimension 2 (mps_chain.uasm) ─────
    print("\n  MPS chain, 256 sites, chi = 2  —  same shape as mps_chain.uasm")
    print("  " + "-" * 62)
    sites = 256
    # ZIPPER2 retires 256 flops: 16 complex MACs to build the intermediate and
    # 16 more to close it against the conjugated bra, 8 flops each.
    flops = sites * 256

    ket = (np.random.randn(sites, 2, 2, 2) + 1j * np.random.randn(sites, 2, 2, 2)
           ).astype(np.complex64)
    bra = (np.random.randn(sites, 2, 2, 2) + 1j * np.random.randn(sites, 2, 2, 2)
           ).astype(np.complex64)

    def mps_cpu():
        e = np.zeros((2, 2), dtype=np.complex64)
        e[0, 0] = 1
        for i in range(sites):
            t = np.einsum("ab,bdc->adc", e, ket[i])
            e = np.einsum("adx,adc->xc", ket[i].conj(), t)
        return e

    record("MPS chain 256 sites", "CPU complex64", timed(mps_cpu, 5), flops)

    if HAS_CUDA:
        kg = torch.from_numpy(ket).cuda()

        def mps_gpu():
            e = torch.zeros(2, 2, dtype=torch.complex64, device="cuda")
            e[0, 0] = 1
            for i in range(sites):
                t = torch.einsum("ab,bdc->adc", e, kg[i])
                e = torch.einsum("adx,adc->xc", kg[i].conj(), t)
            return e

        record("MPS chain 256 sites", "GPU complex64",
               timed(mps_gpu, 5, sync), flops)
        del kg
        torch.cuda.empty_cache()

    os.makedirs("docs", exist_ok=True)
    with open("docs/host-reference.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote docs/host-reference.json")
    print("\nNote: the MPS chain is 256 dependent 2x2 contractions. Neither host")
    print("can parallelise across sites, and on the GPU every site is a separate")
    print("kernel launch, so that row measures launch latency more than arithmetic.")


if __name__ == "__main__":
    main()
