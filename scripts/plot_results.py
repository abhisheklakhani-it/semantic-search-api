"""Draw the README charts from results/ (light and dark versions).

  python scripts/plot_results.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import csv  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, DOCS = ROOT / "results", ROOT / "docs"

# Same validated, colour-blind-safe palette as retrieval-eval-bench: one colour per retrieval method.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "bm25": "#2a78d6", "dense": "#eb6834", "hybrid": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#383835",
             "bm25": "#3987e5", "dense": "#d95926", "hybrid": "#199e70"},
}
METHOD_LABEL = {"bm25": "BM25", "dense": "Dense (BGE-small)", "hybrid": "Hybrid RRF"}
MARKER = {"bm25": "o", "dense": "s", "hybrid": "D"}


def read(name: str) -> list[dict]:
    with (RESULTS / f"{name}.csv").open() as f:
        return list(csv.DictReader(f))


def method(config: str) -> str:
    return "hybrid" if config.startswith("Hybrid") else "dense" if config.startswith("Dense") else "bm25"


def style_axes(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    ax.grid(color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def legend(ax, t: dict, loc="lower left", anchor=(0, 1.0)) -> None:
    handles = [plt.Line2D([], [], marker=MARKER[m], linestyle="", markersize=8, color=t[m], label=METHOD_LABEL[m])
               for m in ["bm25", "dense", "hybrid"]]
    handles += [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=t["surface"],
                   markeredgecolor=t["muted"], markeredgewidth=1.8, label="retrieval only"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=t["muted"], label="+ reranker"),
    ]
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=5, frameon=False, fontsize=9.5,
              labelcolor=t["text"], handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)


def point(ax, t, x, y, m, reranked, size=10):
    ax.plot(x, y, marker=MARKER[m], markersize=size, linestyle="",
            markerfacecolor=t[m] if reranked else t["surface"], markeredgecolor=t[m], markeredgewidth=2, zorder=3)


def is_reranked(config: str) -> bool:
    return "rerank" in config


def tradeoff(theme: str) -> None:
    """nDCG@10 against median latency: what does the reranker buy, and what does it cost?"""
    t = THEMES[theme]
    quality = {r["config"]: float(r["ndcg@10"]) for r in read("quality")}
    latency = {r["config"]: float(r["p50_ms"]) for r in read("latency")}

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200, facecolor=t["surface"])
    style_axes(ax, t)
    reranked = [c for c in quality if is_reranked(c)]
    chain = ["Hybrid RRF", *reranked]  # hybrid alone, then with more and more reranked candidates
    ax.plot([latency[c] for c in chain], [quality[c] for c in chain], color=t["hybrid"], linewidth=1.5, alpha=0.5)
    for config in quality:
        m, x, y = method(config), latency[config], quality[config]
        point(ax, t, x, y, m, is_reranked(config))
        name = config.replace("Hybrid + rerank ", "rerank\n") if is_reranked(config) else config
        ax.annotate(f"{name}\n{y:.3f}", (x, y), xytext=(0, -14), textcoords="offset points",
                    ha="center", va="top", color=t["text"], fontsize=9, linespacing=1.3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
    xs = list(latency.values())
    ax.set_xlim(min(xs) / 1.8, max(xs) * 1.8)
    ys = list(quality.values())
    ax.set_ylim(min(ys) - 0.045, max(ys) + 0.02)
    ax.set_xlabel("Median latency per search (log scale), measured through the HTTP API", color=t["muted"], fontsize=9.5)
    ax.set_ylabel("nDCG@10 (higher is better)", color=t["muted"], fontsize=9.5)
    ax.set_title("On SciFact, reranking cost 25-90x the latency and lowered nDCG@10",
                 loc="left", color=t["text"], fontsize=12, fontweight="bold", pad=30)
    legend(ax, t)
    fig.tight_layout()
    fig.savefig(DOCS / f"tradeoff_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def latency_plot(theme: str) -> None:
    """p50 (marker) and p95 (end of the line) per configuration."""
    t = THEMES[theme]
    rows = read("latency")
    rows.sort(key=lambda r: float(r["p50_ms"]))

    fig, ax = plt.subplots(figsize=(9, 3.9), dpi=200, facecolor=t["surface"])
    style_axes(ax, t)
    ax.grid(axis="y", visible=False)
    for y, r in enumerate(rows):
        m, p50, p95 = method(r["config"]), float(r["p50_ms"]), float(r["p95_ms"])
        ax.plot([p50, p95], [y, y], color=t[m], linewidth=2, solid_capstyle="round", alpha=0.55)
        ax.plot([p95], [y], marker="|", markersize=10, color=t[m], markeredgewidth=2)
        point(ax, t, p50, y, m, is_reranked(r["config"]), size=9)
        ax.text(p95 * 1.08, y, f"p50 {p50:.0f} ms · p95 {p95:.0f} ms", va="center", fontsize=9.5, color=t["text"])
    ax.set_yticks(range(len(rows)), [r["config"] for r in rows], color=t["text"], fontsize=10)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
    ax.set_xlim(min(float(r["p50_ms"]) for r in rows) / 1.5, max(float(r["p95_ms"]) for r in rows) * 4)
    ax.set_xlabel("Latency per search (log scale) · marker = median (p50), bar end = p95", color=t["muted"], fontsize=9.5)
    ax.set_title("Search latency through the API, one request at a time", loc="left", color=t["text"],
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(DOCS / f"latency_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    for theme in THEMES:
        tradeoff(theme)
        latency_plot(theme)
    print("Saved charts to", DOCS.relative_to(ROOT))


if __name__ == "__main__":
    main()
