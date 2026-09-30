"""Benchmark the running API: search quality (nDCG, recall) and latency (p50/p95), with and without the reranker.

Uses SciFact from BEIR (5,183 scientific abstracts, 300 test queries with human relevance labels),
the same dataset as the retrieval-eval-bench project. Everything goes through the HTTP API, so the
numbers include what a real client sees: network, JSON, validation, model inference and Qdrant.

Start the stack with a separate collection and chunks big enough to keep each abstract whole:

  SEMSEARCH_COLLECTION=scifact SEMSEARCH_CHUNK_SIZE=2000 docker compose up -d --build
  python scripts/benchmark.py --url http://localhost:8000
"""

import argparse
import asyncio
import csv
import io
import json
import platform
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

# (label, mode, rerank candidates; 0 = no reranking)
CONFIGS = [
    ("BM25", "bm25", 0),
    ("Dense (BGE-small)", "dense", 0),
    ("Hybrid RRF", "hybrid", 0),
    ("Hybrid + rerank top 10", "hybrid", 10),
    ("Hybrid + rerank top 20", "hybrid", 20),
    ("Hybrid + rerank top 30", "hybrid", 30),
]
LOAD_TEST = [("Hybrid RRF", "hybrid", 0), ("Hybrid + rerank top 20", "hybrid", 20)]


# ---------------------------------------------------------------- data


def load_scifact() -> tuple[list[dict], dict[str, str], dict[str, dict[str, int]]]:
    folder = DATA / "scifact"
    if not folder.exists():
        print("Downloading SciFact (3 MB) ...")
        DATA.mkdir(exist_ok=True)
        zipfile.ZipFile(io.BytesIO(httpx.get(BEIR_URL, timeout=120).content)).extractall(DATA)
    corpus = [json.loads(line) for line in (folder / "corpus.jsonl").open()]
    queries = {q["_id"]: q["text"] for q in map(json.loads, (folder / "queries.jsonl").open())}
    qrels: dict[str, dict[str, int]] = {}
    for line in list((folder / "qrels" / "test.tsv").open())[1:]:
        qid, doc_id, score = line.split()
        if int(score) > 0:
            qrels.setdefault(qid, {})[doc_id] = int(score)
    return corpus, {qid: queries[qid] for qid in qrels}, qrels


def join_title(title: str, text: str) -> str:
    title = title.strip()
    if not title:
        return text
    return f"{title} {text}" if title[-1] in ".?!" else f"{title}. {text}"


# ---------------------------------------------------------------- metrics (as in retrieval-eval-bench)


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    dcg = sum(relevant.get(d, 0) / np.log2(r + 1) for r, d in enumerate(ranked[:k], 1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(r + 1) for r, g in enumerate(ideal, 1))
    return float(dcg / idcg) if idcg else 0.0


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & relevant.keys()) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    return next((1.0 / r for r, d in enumerate(ranked[:k], 1) if d in relevant), 0.0)


def paired_randomization_test(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value that mean(a - b) != 0, by randomly flipping the sign of each per-query difference."""
    diffs = np.asarray(a) - np.asarray(b)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, len(diffs)))
    extreme = (np.abs((signs * diffs).mean(axis=1)) >= abs(diffs.mean()) - 1e-12).sum()
    return float((extreme + 1) / (n + 1))


def bootstrap_ci(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    diffs = np.asarray(a) - np.asarray(b)
    means = diffs[np.random.default_rng(seed).integers(0, len(diffs), size=(n, len(diffs)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


# ---------------------------------------------------------------- API calls


def ingest(client: httpx.Client, corpus: list[dict], batch: int = 64) -> dict:
    stored = {s["source"] for s in client.get("/sources", timeout=120).json()}
    todo = [d for d in corpus if d["_id"] not in stored]  # resumable: skip what an earlier run stored
    if not todo:
        print(f"All {len(corpus)} documents already stored, skipping ingest")
        return {}
    print(f"Ingesting {len(todo)} abstracts ({len(stored)} already stored) ...")
    start = time.perf_counter()
    for i in range(0, len(todo), batch):
        docs = [{"source": d["_id"], "text": join_title(d.get("title", ""), d["text"])} for d in todo[i : i + batch]]
        client.post("/ingest/text", json={"documents": docs}, timeout=600).raise_for_status()
        print(f"  {min(i + batch, len(todo))}/{len(todo)}", end="\r")
    seconds = time.perf_counter() - start
    chunks = client.get("/health").json()["chunks"]
    print(f"\nIngested {len(todo)} docs in {seconds:.1f} s; the collection holds {chunks} chunks")
    return {"docs": len(todo), "seconds": round(seconds, 1), "docs_per_s": round(len(todo) / seconds, 2)}


def params(query: str, mode: str, candidates: int) -> dict:
    p = {"q": query, "k": 10, "mode": mode, "rerank": str(candidates > 0).lower()}
    return p | {"candidates": candidates} if candidates else p


def search(client: httpx.Client, query: str, mode: str, candidates: int) -> tuple[list[str], float, dict]:
    start = time.perf_counter()
    response = client.get("/search", params=params(query, mode, candidates))
    latency = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    body = response.json()
    ranked: list[str] = []
    for hit in body["results"]:  # chunks -> documents, keeping the first (best) chunk of each
        if hit["source"] not in ranked:
            ranked.append(hit["source"])
    return ranked, latency, body["timings"]


async def load_test(url: str, queries: list[str], mode: str, candidates: int, concurrency: int) -> dict:
    """Send all queries with `concurrency` requests in flight at once; report latency and throughput."""
    latencies: list[float] = []
    queue = list(queries)

    async def worker(client: httpx.AsyncClient):
        while queue:
            q = queue.pop()
            start = time.perf_counter()
            r = await client.get("/search", params=params(q, mode, candidates))
            r.raise_for_status()
            latencies.append((time.perf_counter() - start) * 1000)

    async with httpx.AsyncClient(base_url=url, timeout=120) as client:
        start = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        seconds = time.perf_counter() - start
    return {
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "mean_ms": statistics.mean(latencies),
        "qps": len(latencies) / seconds,
    }


# ---------------------------------------------------------------- main


def md_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else (f"{v:.1f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(":--" if i == 0 else "--:" for i in range(len(cols))) + "|"]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=4, help="requests in flight for the load test")
    parser.add_argument("--load-queries", type=int, default=100, help="queries sent in the load test")
    args = parser.parse_args()

    corpus, queries, qrels = load_scifact()
    client = httpx.Client(base_url=args.url, timeout=120)
    ingest_stats = ingest(client, corpus)

    qids = sorted(queries)
    for q in qids[:20]:  # warm-up: first calls pay one-time costs (caches, connection setup)
        search(client, queries[q], "hybrid", 20)

    quality, latency, per_query = [], [], {"query_id": qids}
    for label, mode, candidates in CONFIGS:
        print(f"Running {label} ...")
        runs, lats, stages = {}, [], {"embed_ms": [], "retrieve_ms": [], "rerank_ms": []}
        for qid in qids:
            runs[qid], ms, timings = search(client, queries[qid], mode, candidates)
            lats.append(ms)
            for stage in stages:
                stages[stage].append(timings[stage])
        ndcg = [ndcg_at_k(runs[q], qrels[q]) for q in qids]
        per_query[label] = ndcg
        quality.append(
            {
                "config": label,
                "recall@1": statistics.mean(recall_at_k(runs[q], qrels[q], 1) for q in qids),
                "recall@5": statistics.mean(recall_at_k(runs[q], qrels[q], 5) for q in qids),
                "recall@10": statistics.mean(recall_at_k(runs[q], qrels[q], 10) for q in qids),
                "mrr@10": statistics.mean(reciprocal_rank(runs[q], qrels[q]) for q in qids),
                "ndcg@10": statistics.mean(ndcg),
            }
        )
        latency.append(
            {
                "config": label,
                "p50_ms": percentile(lats, 50),
                "p95_ms": percentile(lats, 95),
                "mean_ms": statistics.mean(lats),
                "embed_p50_ms": percentile(stages["embed_ms"], 50),
                "retrieve_p50_ms": percentile(stages["retrieve_ms"], 50),
                "rerank_p50_ms": percentile(stages["rerank_ms"], 50),
            }
        )

    pairs = [
        ("Hybrid RRF", "Dense (BGE-small)"),
        ("Hybrid RRF", "BM25"),
        ("Hybrid + rerank top 10", "Hybrid RRF"),
        ("Hybrid + rerank top 20", "Hybrid RRF"),
        ("Hybrid + rerank top 30", "Hybrid RRF"),
        ("Hybrid + rerank top 20", "Hybrid + rerank top 10"),
        ("Hybrid + rerank top 30", "Hybrid + rerank top 20"),
    ]
    significance = []
    for a, b in pairs:
        low, high = bootstrap_ci(per_query[a], per_query[b])
        p = paired_randomization_test(per_query[a], per_query[b])
        significance.append(
            {
                "A": a,
                "B": b,
                "ndcg@10 A - B": statistics.mean(per_query[a]) - statistics.mean(per_query[b]),
                "95% CI": f"[{low:+.3f}, {high:+.3f}]",
                "p-value": p,
                "significant": "yes" if p < 0.05 else "no",
            }
        )

    print(f"Load test with {args.concurrency} concurrent requests ...")
    load = []
    for label, mode, candidates in LOAD_TEST:
        subset = [queries[q] for q in qids[: args.load_queries]]
        stats = asyncio.run(load_test(args.url, subset, mode, candidates, args.concurrency))
        load.append({"config": label, "concurrency": args.concurrency, **stats})

    RESULTS.mkdir(exist_ok=True)


    for name, rows in [("quality", quality), ("latency", latency), ("significance", significance), ("load", load)]:
        with (RESULTS / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (RESULTS / "per_query_ndcg.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(per_query.keys())
        writer.writerows(zip(*per_query.values()))

    health = client.get("/health").json()
    docker_cpus = subprocess.run(["docker", "info", "--format", "{{.NCPU}}"], capture_output=True, text=True).stdout.strip()
    config = {
        "dataset": "BEIR SciFact test split",
        "queries": len(qids),
        "documents": len(corpus),
        "chunks": health["chunks"],
        "k": 10,
        "api_url": args.url,
        "machine": f"{platform.machine()} {platform.platform()}",
        "docker_cpus": docker_cpus or None,
        "ingest": ingest_stats or None,
        "date": time.strftime("%Y-%m-%d"),
    }
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2))

    report = "\n\n".join(
        [
            "## Quality (SciFact test, 300 queries)\n\n" + md_table(quality),
            "## Latency, one request at a time (client-side ms; stage columns are server-side p50)\n\n" + md_table(latency),
            "## Load test\n\n" + md_table(load),
            "## Significance (paired randomization test on per-query nDCG@10)\n\n" + md_table(significance),
        ]
    )
    (RESULTS / "results.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    sys.exit(main())
