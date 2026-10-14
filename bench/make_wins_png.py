"""Only the places Vega came out ahead of the hosted reference, drawn from a finished run.

    python make_wins_png.py /path/to/vega_benchmark.json [out.png]

The win list is derived, not typed in: every public task where either engine scored above the
hosted model, every JevBench and DecisionBench slice where it did, and every suite level metric
where it did. If a future run loses one of these, the row disappears by itself.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, BAND = "#e4e3df", "#f4f3f0"

plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.family": "DejaVu Sans", "text.color": INK})

JEV = "jev/jev-latest"
V08, V4 = "vega-0.8b/engine", "vega-4b/engine"
EPS = 1e-9


def pub_macro(d, name, field):
    g = [rs[name][field] for rs in d["results"]["public"].values()
         if name in rs and rs[name].get(field) is not None]
    if not g:
        return None
    return float(np.median(g)) if field.startswith("latency") else float(np.mean(g))


def collect(d):
    """(band, label, v08, v4, jev, fmt, direction) for everything Vega wins."""
    res, pub = d["results"], d["results"]["public"]
    rows = []

    won = []
    for key in sorted(pub):
        rs = pub[key]
        j = rs.get(JEV, {}).get("accuracy")
        if j is None:
            continue
        a = [rs.get(V08, {}).get("accuracy"), rs.get(V4, {}).get("accuracy")]
        best = max([x for x in a if x is not None] or [None])
        if best is None or best <= j + EPS:
            continue
        n = rs["_meta"]["n_eval"]
        won.append((round((best - j) * n), key, n, a[0], a[1], j))
    for netrows, key, n, v08, v4, j in sorted(won, reverse=True):
        rows.append(("Public tasks it wins, of the 43 on the menu",
                     "%s, %d rows, %+d" % (key, n, netrows), v08, v4, j, "%.3f", "hi"))

    for suite, title in (("jevbench", "JevBench"), ("decisionbench", "DecisionBench")):
        for sl in sorted(res[suite]):
            if sl in ("all", "paraphrase", "soft_gold") or sl.startswith("task:"):
                continue
            blk = res[suite][sl]
            j = blk.get(JEV, {}).get("accuracy")
            if j is None:
                continue
            a = [blk.get(V08, {}).get("accuracy"), blk.get(V4, {}).get("accuracy")]
            best = max([x for x in a if x is not None] or [None])
            if best is None or best <= j + EPS:
                continue
            n = blk.get(V4, {}).get("n") or blk.get(V08, {}).get("n")
            rows.append(("Slices it wins", "%s %s, %s items" % (title, sl, n),
                         a[0], a[1], j, "%.3f", "hi"))

    metrics = [
        ("public", "nll", "lo", "Public menu, negative log likelihood", "%.3f"),
        ("public", "latency_ms_p50", "lo", "Public menu, p50 ms per decision", "%.0f"),
        ("decisionbench", "ece", "lo", "DecisionBench, calibration error", "%.3f"),
        ("decisionbench", "nll", "lo", "DecisionBench, negative log likelihood", "%.3f"),
        ("decisionbench", "soft_gold", "lo", "DecisionBench, KL to the gold distribution", "%.3f"),
        ("decisionbench", "coverage", "hi", "DecisionBench, rows answered", "%.3f"),
        ("decisionbench", "latency_ms_p50", "lo", "DecisionBench, p50 ms per decision", "%.0f"),
    ]
    for source, field, direction, label, f in metrics:
        if source == "public":
            vals = [pub_macro(d, n, field) for n in (V08, V4, JEV)]
        elif field == "soft_gold":
            sg = res[source].get("soft_gold", {})
            vals = [sg.get(n, {}).get("soft_gold_kl_mean") for n in (V08, V4, JEV)]
        else:
            vals = [res[source]["all"].get(n, {}).get(field) for n in (V08, V4, JEV)]
        v08, v4, j = vals
        if j is None:
            continue
        better = [v for v in (v08, v4) if v is not None
                  and (v < j - EPS if direction == "lo" else v > j + EPS)]
        if better:
            rows.append(("Calibration and cost, whole suites", label, v08, v4, j, f,
                         direction))
    return rows


def main(path, out):
    d = json.load(open(path))
    rows = collect(d)
    bands = []
    for band, label, v08, v4, j, f, direction in rows:
        if not bands or bands[-1][0] != band:
            bands.append((band, []))
        bands[-1][1].append((label, v08, v4, j, f, direction))

    n_data = len(rows)
    n_band = len(bands)
    fig_h = 1.72 + n_data * 0.43 + n_band * 0.67 + 1.05
    fig = plt.figure(figsize=(11.6, fig_h), dpi=190)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    x_label, x_cols = 3.0, [56.0, 72.5, 89.0]
    top = fig_h - 1.26

    fig.text(x_label / 100.0, (fig_h - 0.42) / fig_h,
             "Where Vega comes out ahead of the hosted model",
             fontsize=15.5, fontweight="bold", va="center")
    fig.text(x_label / 100.0, (fig_h - 0.76) / fig_h,
             "One run, 2026-10-11, two Tesla T4s, 3,841 decisions per system. "
             "Only the rows Vega wins. The rest of the menu goes the other way.",
             fontsize=9.4, color=INK2, va="center")

    for x, head, sub in zip(x_cols, ("Vega 0.8B", "Vega 4B", "Jev"),
                            ("local, 0.8B", "local, 4B", "hosted, jev-latest"), strict=True):
        ax.text(x, top, head, fontsize=11.6, fontweight="bold", ha="center", va="center",
                color=ORANGE if head == "Jev" else BLUE)
        ax.text(x, top - 0.23, sub, fontsize=8.0, ha="center", va="center", color=INK3)
    ax.plot([x_label - 0.6, 97], [top - 0.40, top - 0.40], color=INK, lw=1.2)

    y = top - 0.40
    for band, items in bands:
        y -= 0.50
        ax.add_patch(plt.Rectangle((x_label - 0.6, y - 0.15), 97 - x_label + 0.6, 0.38,
                                   facecolor=BAND, edgecolor="none", zorder=0, clip_on=False))
        ax.text(x_label, y + 0.04, band, fontsize=9.6, fontweight="bold", va="center", color=INK2)
        y -= 0.17
        for label, v08, v4, j, f, direction in items:
            y -= 0.43
            ax.text(x_label, y, label, fontsize=10.2, va="center", color=INK)
            vals = [v for v in (v08, v4, j) if v is not None]
            best = (min if direction == "lo" else max)(vals) if vals else None
            for x, v in zip(x_cols, (v08, v4, j), strict=True):
                if v is None:
                    ax.text(x, y, "not run", fontsize=10.0, ha="center", va="center", color=INK3)
                    continue
                win = abs(v - best) < 1e-12
                colour = (ORANGE if x == x_cols[2] else BLUE) if win else INK2
                ax.text(x, y, f % v, fontsize=10.6, ha="center", va="center", color=colour,
                        fontweight="bold" if win else "normal")
            ax.plot([x_label - 0.6, 97], [y - 0.215, y - 0.215], color=GRID, lw=0.7, zorder=0)

    env = d["environment"]
    ax.text(x_label, y - 0.60,
            "A task label carries its row count and the win in rows: +1 is one row of sixty. "
            "Eight of the twelve task wins are three rows or fewer, and two clear six.\n"
            "vegaml %s, torch %s, transformers %s. Reproduce with notebooks/vega_benchmark_2xT4.ipynb."
            % (env.get("vegaml"), env.get("torch"), env.get("transformers")),
            fontsize=8.0, color=INK3, va="center", linespacing=1.5)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.26)
    print("wrote %s: %d rows in %d bands" % (out, n_data, n_band))


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "vega_benchmark_wins.png")
    main(src, dst)
