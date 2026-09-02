"""Render the benchmark data in docs/bench.csv as SVG charts.

No dependencies: the SVG is written directly. Light theme on purpose — these are
measurements, meant to be readable on a white page, in a PDF and on paper.

Run from the repository root, after `unibit bench`:

    python tools/plot_bench.py
"""
import csv
import io
import os

# ── Light theme ────────────────────────────────────────────────────────────
BG      = "#ffffff"
INK     = "#1b1f23"   # primary text
MUTED   = "#6a737d"   # axis labels, captions
GRID    = "#e6e9ec"
AXIS    = "#c4cad0"
SERIES1 = "#2f6f9f"   # measured / primary
SERIES2 = "#c96a2b"   # baseline / comparison
GOOD    = "#3d8b5f"
FONT    = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def header(w, h, title, subtitle):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"
     viewBox="0 0 {w} {h}" font-family="{FONT}">
  <rect width="{w}" height="{h}" fill="{BG}"/>
  <text x="28" y="34" font-size="17" font-weight="600" fill="{INK}">{esc(title)}</text>
  <text x="28" y="55" font-size="12.5" fill="{MUTED}">{esc(subtitle)}</text>
"""


def grouped_bars(rows, path):
    """Branch prediction accuracy: BHT/BTB against static not-taken."""
    W, H = 860, 420
    L, R, T, B = 90, 30, 90, 96
    pw, ph = W - L - R, H - T - B
    n = len(rows)
    slot = pw / n
    bw = min(30, slot * 0.30)

    s = header(W, H, "Branch prediction accuracy",
               "512-entry BHT/BTB with 2-bit saturating counters, against a static "
               "not-taken baseline. Same program, same input; only the policy differs.")

    for i in range(0, 101, 25):                      # gridlines + y axis
        y = T + ph - (i / 100) * ph
        s += (f'  <line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{GRID}"/>\n'
              f'  <text x="{L-12}" y="{y+4:.1f}" font-size="11.5" fill="{MUTED}" '
              f'text-anchor="end">{i}%</text>\n')

    for i, r in enumerate(rows):
        x0 = L + i * slot + slot / 2
        for j, (key, colour) in enumerate((("accuracy_bht", SERIES1),
                                           ("accuracy_static", SERIES2))):
            v = float(r[key])
            h = (v / 100) * ph
            x = x0 - bw - 3 + j * (bw + 6)
            s += (f'  <rect x="{x:.1f}" y="{T+ph-h:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                  f'fill="{colour}" rx="2"/>\n')
            if h > 26:
                s += (f'  <text x="{x+bw/2:.1f}" y="{T+ph-h+15:.1f}" font-size="10.5" '
                      f'fill="{BG}" text-anchor="middle">{v:.0f}</text>\n')
        s += (f'  <text x="{x0:.1f}" y="{T+ph+20:.1f}" font-size="11" fill="{INK}" '
              f'text-anchor="middle">{esc(r["program"])}</text>\n'
              f'  <text x="{x0:.1f}" y="{T+ph+35:.1f}" font-size="10" fill="{MUTED}" '
              f'text-anchor="middle">{int(r["branches"]):,} br</text>\n')

    s += f'  <line x1="{L}" y1="{T+ph}" x2="{W-R}" y2="{T+ph}" stroke="{AXIS}"/>\n'
    for k, (lbl, c) in enumerate((("BHT/BTB", SERIES1), ("static not-taken", SERIES2))):
        x = L + k * 150
        s += (f'  <rect x="{x}" y="{H-42}" width="11" height="11" fill="{c}" rx="2"/>\n'
              f'  <text x="{x+17}" y="{H-32}" font-size="11.5" fill="{INK}">{lbl}</text>\n')
    s += (f'  <text x="{W-R}" y="{H-32}" font-size="11" fill="{MUTED}" text-anchor="end">'
          f'higher is better</text>\n</svg>\n')
    io.open(path, "w", encoding="utf-8").write(s)
    print("wrote", path)


def throughput(rows, path):
    """Emulator throughput in million instructions per second."""
    W, H = 860, 380
    L, R, T, B = 90, 30, 90, 70
    pw, ph = W - L - R, H - T - B
    n = len(rows)
    slot = pw / n
    bw = min(46, slot * 0.42)
    peak = max(float(r["mips"]) for r in rows)
    top = max(1.0, peak * 1.15)

    s = header(W, H, "Emulator throughput",
               "Million instructions retired per second, mean of 50 runs with output "
               "captured. Debug-free release build, single thread.")

    for i in range(5):
        v = top * i / 4
        y = T + ph - (v / top) * ph
        s += (f'  <line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{GRID}"/>\n'
              f'  <text x="{L-12}" y="{y+4:.1f}" font-size="11.5" fill="{MUTED}" '
              f'text-anchor="end">{v:.0f}</text>\n')

    for i, r in enumerate(rows):
        v = float(r["mips"])
        h = (v / top) * ph
        x = L + i * slot + slot / 2 - bw / 2
        s += (f'  <rect x="{x:.1f}" y="{T+ph-h:.1f}" width="{bw:.1f}" height="{h:.1f}" '
              f'fill="{SERIES1}" rx="2"/>\n'
              f'  <text x="{x+bw/2:.1f}" y="{T+ph-h-7:.1f}" font-size="11" fill="{INK}" '
              f'text-anchor="middle">{v:.1f}</text>\n'
              f'  <text x="{x+bw/2:.1f}" y="{T+ph+20:.1f}" font-size="11" fill="{INK}" '
              f'text-anchor="middle">{esc(r["program"])}</text>\n')

    s += (f'  <line x1="{L}" y1="{T+ph}" x2="{W-R}" y2="{T+ph}" stroke="{AXIS}"/>\n'
          f'  <text x="28" y="{T-14}" font-size="11.5" fill="{MUTED}">MIPS</text>\n'
          f'  <text x="{W-R}" y="{H-22}" font-size="11" fill="{MUTED}" text-anchor="end">'
          f'short programs are dominated by process and assembly overhead</text>\n</svg>\n')
    io.open(path, "w", encoding="utf-8").write(s)
    print("wrote", path)


def error_budget(path):
    """Mixed-precision error budget of the chi = 2 zipper, on a log scale.

    Values come from test_zipper2_error_budget_is_dominated_by_the_cores,
    which runs the ablation over 200 random chi = 2 MPS of 8 sites.
    """
    items = [("int8 cores — norm drift (worst)",  1.391e-2, SERIES2),
             ("int8 cores — norm drift (mean)",   2.484e-3, SERIES2),
             ("int8 cores — 1 − fidelity (worst)", 7.9e-5,  SERIES1),
             ("f32 accumulator — rel. error (worst)", 5.002e-7, GOOD)]

    W, H = 860, 330
    L, R, T, B = 300, 60, 92, 66
    pw, ph = W - L - R, H - T - B
    import math
    lo, hi = -7.5, -1.5                              # log10 decades shown

    s = header(W, H, "Mixed-precision error budget, chi = 2 zipper",
               "Ablation over 200 random bond-dimension-2 MPS of 8 sites. "
               "The int8 cores carry the entire loss; the f32 accumulator is free.")

    for d in range(int(lo) + 1, int(hi) + 1):        # decade gridlines
        x = L + (d - lo) / (hi - lo) * pw
        s += (f'  <line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}" stroke="{GRID}"/>\n'
              f'  <text x="{x:.1f}" y="{T+ph+20}" font-size="11" fill="{MUTED}" '
              f'text-anchor="middle">1e{d}</text>\n')

    rh = ph / len(items)
    for i, (label, v, colour) in enumerate(items):
        y = T + i * rh + rh / 2
        x = L + (math.log10(v) - lo) / (hi - lo) * pw
        s += (f'  <line x1="{L}" y1="{y:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
              f'stroke="{colour}" stroke-width="2.5" stroke-linecap="round"/>\n'
              f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colour}"/>\n'
              f'  <text x="{L-16}" y="{y+4:.1f}" font-size="12" fill="{INK}" '
              f'text-anchor="end">{esc(label)}</text>\n'
              f'  <text x="{x+12:.1f}" y="{y+4:.1f}" font-size="11" fill="{MUTED}">'
              f'{v:.3g}</text>\n')

    s += (f'  <line x1="{L}" y1="{T+ph}" x2="{W-R}" y2="{T+ph}" stroke="{AXIS}"/>\n'
          f'  <text x="{W-R}" y="{H-20}" font-size="11" fill="{MUTED}" text-anchor="end">'
          f'relative error, log scale — lower is better</text>\n</svg>\n')
    io.open(path, "w", encoding="utf-8").write(s)
    print("wrote", path)


def main():
    if not os.path.exists("docs/bench.csv"):
        raise SystemExit("docs/bench.csv not found — run `unibit bench` first.")
    with io.open("docs/bench.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    os.makedirs("docs/img", exist_ok=True)
    grouped_bars(rows, "docs/img/branch-prediction.svg")
    throughput(rows, "docs/img/throughput.svg")
    error_budget("docs/img/error-budget.svg")


if __name__ == "__main__":
    main()
