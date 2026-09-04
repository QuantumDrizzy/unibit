"""Arithmetic density chart: what each kernel costs this ISA, against its ceiling.

The other plots in tools/ emit hand-written SVG so the repository stays free of
dependencies. GitHub will not render an SVG served from raw.githubusercontent,
so this one uses matplotlib to emit a PNG for the README and the profile page.
It is a development tool; the crate does not know it exists.

    python tools/plot_density.py        # -> docs/img/density.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Light theme, the same palette the rest of the ecosystem's figures use.
BG, TEXT, GRID = "#ffffff", "#1a1d21", "#d8dce0"
ACHIEVED, CEILING, REFLINE = "#0b6ea8", "#c9d6e0", "#c0392b"

# From docs/kernel-cost.md. Ceilings are the loop body that cannot be removed:
# VDOT.B kernels need two lq, one VDOT.B and one add per 32 int8 MACs
# (64 flops / 4 instructions = 16.0); ZIPPER2 needs two lq and one ZIPPER2,
# with no accumulate because rd is the accumulator (256 / 3 = 85.3).
KERNELS = [
    ("llm_matvec\nint8 matvec 64x3584", 13.60, 16.0),
    ("ising_energy\ncoupling energy n=256", 10.48, 16.0),
    ("mps_chain\n256-site MPS, chi=2", 69.57, 85.3),
]


def main():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
        "grid.color": GRID, "font.family": "DejaVu Sans Mono", "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    xs = range(len(KERNELS))
    labels = [k[0] for k in KERNELS]
    achieved = [k[1] for k in KERNELS]
    ceilings = [k[2] for k in KERNELS]

    ax.bar(xs, ceilings, width=0.55, color=CEILING, zorder=2,
           label="ISA ceiling (irreducible loop body)")
    ax.bar(xs, achieved, width=0.55, color=ACHIEVED, zorder=3,
           label="measured")

    for x, (a, c) in enumerate(zip(achieved, ceilings)):
        ax.text(x, a - c * 0.06, f"{a:.2f}", ha="center", va="top",
                color="#ffffff", fontsize=11, fontweight="bold", zorder=4)
        ax.text(x, c + 1.2, f"{a / c * 100:.0f}% of {c:g}", ha="center",
                va="bottom", color=TEXT, fontsize=9)

    # AVX2's VPMADDUBSW does 32 int8 MACs over a 256-bit YMM register: the same
    # 64 flops per instruction as VDOT.B, so the same 16.0 kernel ceiling. Width
    # is not what buys density here - only ZIPPER2's fusion breaks past it.
    ax.axhline(16.0, color=REFLINE, linestyle="--", linewidth=1.3, zorder=5)
    ax.text(-0.42, 22.0, "AVX2 VPMADDUBSW ceiling — identical at 16.0.\n"
                         "256-bit width buys nothing; only ZIPPER2's fusion does.",
            ha="left", va="bottom", color=REFLINE, fontsize=9, style="italic")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("floating-point operations retired per instruction")
    ax.set_title("Unibit — what three real workloads cost this ISA\n"
                 "Emulated, so only flops/instruction is architectural. "
                 "Wall-clock would measure the emulator.",
                 fontsize=11, fontweight="bold", pad=14)
    ax.set_ylim(0, 96)
    ax.grid(axis="y", alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out = "docs/img/density.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
