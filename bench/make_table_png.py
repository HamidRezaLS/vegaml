"""The headline table as one image, drawn straight from a finished run's JSON.

    python make_table_png.py /path/to/vega_benchmark.json [out.png]

Nothing here is typed in by hand: every cell is read from the results file, so the picture and
the tables in README.md cannot drift apart. The winning cell in each row is coloured and bold,
and the arrow after a metric name says which direction wins.
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

plt.rcParams.update({
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK,
})

JEV = "jev/jev-latest"
V08, V4 = "vega-0.8b/engine", "vega-4b/engine"
COLS = [(V08, "Vega 0.8B", "local, 0.8B"), (V4, "Vega 4B", "local, 4B"),
        (JEV, "Jev", "hosted, jev-latest")]

# (label, direction, source, field). direction "hi" or "lo"; "-" rows are group headers.
LAYOUT = [
    ("Public menu, 43 tasks, macro average over tasks", None, None, None),
    ("accuracy", "hi", "public", "accuracy"),
    ("skill, chance corrected", "hi", "public", "skill"),
    ("macro F1", "hi", "public", "macro_f1"),
    ("expected calibration error", "lo", "public", "ece"),
    ("negative log likelihood", "lo", "public", "nll"),
    ("area under the risk coverage curve", "lo", "public", "aurc"),
    ("latency per decision, p50 ms", "lo", "public", "latency_ms_p50"),
    ("JevBench, 231 items", None, None, None),
    ("accuracy", "hi", "jevbench", "accuracy"),
    ("accuracy, hard tier, 111 items", "hi", "jevbench", "tier:hard"),
    ("paraphrase consistency, 36 groups", "hi", "jevbench", "paraphrase"),
    ("expected calibration error", "lo", "jevbench", "ece"),
    ("DecisionBench, 1,075 items", None, None, None),
    ("accuracy", "hi", "decisionbench", "accuracy"),
    ("expected calibration error", "lo", "decisionbench", "ece"),
    ("negative log likelihood", "lo", "decisionbench", "nll"),
    ("KL to the gold distribution, 10 rows", "lo", "decisionbench", "soft_gold"),
    ("coverage, rows answered", "hi", "decisionbench", "coverage"),
    ("latency per decision, p50 ms", "lo", "decisionbench", "latency_ms_p50"),
]


def read(d, source, field, name):
    res = d["results"]
    if source == "public":
        g = [rs[name][field] for rs in res["public"].values()
             if name in rs and rs[name].get(field) is not None]
        if not g:
            return None
        return float(np.median(g)) if field.startswith("latency") else float(np.mean(g))
    blk = res[source]
    if field == "paraphrase":
        return blk.get("paraphrase", {}).get(name, {}).get("paraphrase_consistency")
    if field == "soft_gold":
        return blk.get("soft_gold", {}).get(name, {}).get("soft_gold_kl_mean")
    if field.startswith("tier:"):
        return blk.get(field, {}).get(name, {}).get("accuracy")
    return blk["all"].get(name, {}).get(field)


def fmt(v, field):
    if v is None:
        return "not run"
    return "%.0f" % v if field.startswith("latency") else "%.3f" % v


def main(path, out):
    d = json.load(open(path))
    rows = []
    for label, direction, source, field in LAYOUT:
        if direction is None:
            rows.append((label, None, None, None))
            continue
        vals = [read(d, source, field, name) for name, _, _ in COLS]
        good = [v for v in vals if v is not None]
        best = (max if direction == "hi" else min)(good) if good else None
        cells = [(fmt(v, field), v is not None and best is not None and abs(v - best) < 1e-12)
                 for v in vals]
        rows.append((label + ("  ↑" if direction == "hi" else "  ↓"), cells, None, None))

    n_band = sum(1 for r in rows if r[1] is None)
    fig_h = 1.66 + (len(rows) - n_band) * 0.43 + n_band * 0.67 + 0.95
    fig = plt.figure(figsize=(11.6, fig_h), dpi=190)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    x_label = 3.0
    x_cols = [56.0, 72.5, 89.0]
    top = fig_h - 1.26

    n_rows = (sum(rs["_meta"]["n_eval"] for rs in d["results"]["public"].values())
              + d["results"]["jevbench"]["all"][JEV]["n"]
              + d["results"]["decisionbench"]["all"][JEV]["n"])
    fig.text(x_label / 100.0, (fig_h - 0.42) / fig_h,
             "Two local models and a hosted one, over %s decisions each" % format(n_rows, ","),
             fontsize=15.5, fontweight="bold", va="center")
    fig.text(x_label / 100.0, (fig_h - 0.76) / fig_h,
             "One run, 2026-10-11, two Tesla T4s. Same items, same prompts, same scorer. "
             "The arrow says which way wins.",
             fontsize=9.4, color=INK2, va="center")

    for x, (_, head, sub) in zip(x_cols, COLS):
        colour = ORANGE if head == "Jev" else BLUE
        ax.text(x, top, head, fontsize=11.6, fontweight="bold", ha="center",
                va="center", color=colour)
        ax.text(x, top - 0.23, sub, fontsize=8.0, ha="center", va="center", color=INK3)
    ax.plot([x_label - 0.6, 97], [top - 0.40, top - 0.40], color=INK, lw=1.2)

    y = top - 0.40
    for label, cells, _, _ in rows:
        if cells is None:
            y -= 0.50
            ax.add_patch(plt.Rectangle((x_label - 0.6, y - 0.15), 97 - x_label + 0.6, 0.38,
                                       facecolor=BAND, edgecolor="none", zorder=0,
                                       clip_on=False))
            ax.text(x_label, y + 0.04, label, fontsize=9.6, fontweight="bold",
                    va="center", color=INK2)
            y -= 0.17
            continue
        y -= 0.43
        ax.text(x_label, y, label, fontsize=10.2, va="center", color=INK)
        for x, (text, win) in zip(x_cols, cells):
            colour = INK3 if text == "not run" else (
                (ORANGE if x == x_cols[2] else BLUE) if win else INK2)
            ax.text(x, y, text, fontsize=10.6, ha="center", va="center", color=colour,
                    fontweight="bold" if win else "normal")
        ax.plot([x_label - 0.6, 97], [y - 0.215, y - 0.215], color=GRID, lw=0.7, zorder=0)

    env = d["environment"]
    foot = ("43 public tasks at up to 60 rows each, JevBench public tiers, DecisionBench. "
            "Zero-shot, mode=\"engine\", no tuning on evaluation data.\n"
            "vegaml %s, torch %s, transformers %s. Reproduce with notebooks/vega_benchmark_2xT4.ipynb."
            % (env.get("vegaml"), env.get("torch"), env.get("transformers")))
    ax.text(x_label, y - 0.56, foot, fontsize=8.0, color=INK3, va="center", linespacing=1.5)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.26)
    print("wrote %s" % out)


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "vega_benchmark_table.png")
    main(src, dst)
