#!/usr/bin/env python3
"""Plot single- vs multi-stream pipeline benchmarks.

Reads pipeline_benchmark.csv (written by filter_pipeline.exe) and produces:
  * bench_ms.png       - ms/frame, single vs multi(2/4/8), grouped by resolution
  * bench_speedup.png  - multi-stream speedup relative to single, vs stream count
Also converts the *.ppm visual dumps to *.png if Pillow is available.

Usage:  python plot_benchmarks.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "pipeline_benchmark.csv")


def load():
    rows = []
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            r["pipeline_len"] = int(r["pipeline_len"])
            r["streams"] = int(r["streams"])
            r["ms"] = float(r["ms"])
            r["fps"] = float(r["fps"])
            rows.append(r)
    return rows


def plot_ms(rows):
    resolutions = ["480p", "720p", "1080p"]
    lens = sorted({r["pipeline_len"] for r in rows})
    configs = [("single", 1), ("multi", 2), ("multi", 4), ("multi", 8)]
    labels = ["single", "multi x2", "multi x4", "multi x8"]
    colors = ["#3b7dd8", "#69b076", "#d9a441", "#c0584b"]

    fig, axes = plt.subplots(1, len(resolutions), figsize=(14, 4.2), sharey=False)
    for ax, res in zip(axes, resolutions):
        width = 0.2
        for ci, ((mode, ns), lab, col) in enumerate(zip(configs, labels, colors)):
            ys = []
            for L in lens:
                m = [r["ms"] for r in rows
                     if r["resolution"] == res and r["pipeline_len"] == L
                     and r["mode"] == mode and r["streams"] == ns]
                ys.append(m[0] if m else 0)
            xs = [i + (ci - 1.5) * width for i in range(len(lens))]
            ax.bar(xs, ys, width, label=lab, color=col)
        ax.set_title(res)
        ax.set_xticks(range(len(lens)))
        ax.set_xticklabels([f"{L} stages" for L in lens])
        ax.set_ylabel("ms / frame")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Pipeline latency: single vs multi-stream (RTX 3050 Ti Laptop)")
    fig.tight_layout()
    out = os.path.join(HERE, "bench_ms.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


def plot_speedup(rows):
    resolutions = ["480p", "720p", "1080p"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {"480p": "o", "720p": "s", "1080p": "^"}
    # Use the 3-stage pipeline as the representative chain.
    L = 3
    for res in resolutions:
        single = [r["ms"] for r in rows
                  if r["resolution"] == res and r["pipeline_len"] == L
                  and r["mode"] == "single"][0]
        xs, ys = [], []
        for ns in [2, 4, 8]:
            m = [r["ms"] for r in rows
                 if r["resolution"] == res and r["pipeline_len"] == L
                 and r["mode"] == "multi" and r["streams"] == ns]
            if m:
                xs.append(ns)
                ys.append(single / m[0])
        ax.plot(xs, ys, marker=markers[res], label=res)
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6, label="parity")
    ax.set_xlabel("number of streams")
    ax.set_ylabel("speedup vs single-stream  (>1 = faster)")
    ax.set_title("Multi-stream speedup, 3-stage pipeline")
    ax.set_xticks([2, 4, 8])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "bench_speedup.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


def convert_ppms():
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed; skipping PPM->PNG conversion")
        return
    for name in ("pipeline_input", "pipeline_output", "pipeline_wipe"):
        src = os.path.join(HERE, name + ".ppm")
        if os.path.exists(src):
            Image.open(src).save(os.path.join(HERE, name + ".png"))
            print("wrote", name + ".png")


if __name__ == "__main__":
    rows = load()
    plot_ms(rows)
    plot_speedup(rows)
    convert_ppms()
