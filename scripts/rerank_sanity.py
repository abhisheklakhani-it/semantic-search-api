"""Sanity check: is the reranker wired correctly? Reproduce a published number through the API.

The BEIR paper (Thakur et al., 2021, Table 2 / appendix) reports nDCG@10 = 0.688 on SciFact for
"BM25 + CE": BM25's top 100 reranked by the same cross-encoder (ms-marco-MiniLM-L-6-v2).
This script runs that setting through GET /search (mode=bm25, candidates=100) and prints our score.

  python scripts/rerank_sanity.py --url http://localhost:8000
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import RESULTS, load_scifact, ndcg_at_k, search  # noqa: E402

PUBLISHED = 0.688


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    _, queries, qrels = load_scifact()
    client = httpx.Client(base_url=args.url, timeout=300)
    scores = {}
    for mode, candidates in [("bm25", 0), ("bm25", 100)]:
        ndcg = []
        for i, qid in enumerate(sorted(queries), 1):
            ranked, _, _ = search(client, queries[qid], mode, candidates)
            ndcg.append(ndcg_at_k(ranked, qrels[qid]))
            print(f"  {mode} candidates={candidates}: {i}/{len(queries)}", end="\r")
        scores[f"{mode}, rerank top {candidates}" if candidates else mode] = statistics.mean(ndcg)
        print()
    result = {"ndcg@10": scores, "published_bm25_plus_ce": PUBLISHED}
    (RESULTS / "rerank_sanity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
