from tests.conftest import upload
from tests.pdf_utils import make_pdf

BATTERY = "Lithium batteries age faster at high temperature. Keep the charge between 20 and 80 percent."
RADAR = "Radar measures distance and speed with the Doppler effect and works in fog and rain."
LIDAR = "Lidar builds a precise 3D point cloud with laser pulses but struggles in heavy snow."


def ids(response) -> list[str]:
    return [f"{h['source']}#{h['chunk_index']}" for h in response.json()["results"]]


# ---- health and ops ----


def test_health_reports_qdrant_and_chunk_count(client):
    assert client.get("/health").json() == {"status": "ok", "qdrant": "ok", "collection": "documents", "chunks": 0}
    upload(client, "battery.txt", BATTERY)
    assert client.get("/health").json()["chunks"] == 1


def test_request_id_is_returned_and_can_be_passed_in(client):
    generated = client.get("/health").headers["X-Request-ID"]
    assert len(generated) == 16
    assert client.get("/health", headers={"X-Request-ID": "trace-42"}).headers["X-Request-ID"] == "trace-42"


# ---- ingest ----


def test_ingest_text_file_then_find_it(client):
    r = upload(client, "battery.txt", BATTERY)
    assert r.status_code == 201
    assert r.json()["source"] == "battery.txt" and r.json()["chunks"] == 1 and r.json()["pages"] is None
    upload(client, "radar.txt", RADAR)

    hits = client.get("/search", params={"q": "battery temperature"}).json()["results"]
    assert hits[0]["source"] == "battery.txt"
    assert hits[0]["text"] == BATTERY


def test_ingest_pdf_keeps_page_numbers(client):
    pdf = make_pdf(["Introduction to vehicles.", "Radar sensors work in fog.", "Summary."])
    r = upload(client, "sensors.pdf", pdf)
    assert r.status_code == 201 and r.json()["pages"] == 3
    top = client.get("/search", params={"q": "radar fog", "k": 1}).json()["results"][0]
    assert (top["source"], top["page"]) == ("sensors.pdf", 2)


def test_reingesting_a_source_replaces_it_without_duplicates(make_client):
    client = make_client(chunk_size=100, chunk_overlap=0)
    long_text = ". ".join(f"Sentence number {i} about batteries" for i in range(20))
    first = upload(client, "doc.txt", long_text).json()
    assert first["chunks"] > 3

    second = upload(client, "doc.txt", "Now a short document about batteries.").json()
    assert second["chunks"] == 1
    assert second["removed_old_chunks"] == first["chunks"] - 1
    assert client.get("/sources").json() == [{"source": "doc.txt", "chunks": 1}]


def test_ingest_errors(make_client):
    client = make_client(max_upload_mb=0.001)  # about 1 KB
    assert upload(client, "empty.txt", b"").status_code == 400
    assert upload(client, "table.csv", "a,b").status_code == 415
    assert upload(client, "big.txt", "x" * 2000).status_code == 413
    assert upload(client, "blank.pdf", make_pdf(["", ""])).status_code == 422  # no text layer
    assert upload(client, "broken.pdf", b"%PDF-1.4 not really a pdf").status_code == 400


def test_ingest_text_endpoint_batches_documents(client):
    body = {
        "documents": [
            {"source": "radar", "text": RADAR, "date": "2026-01-10"},
            {"source": "lidar", "text": LIDAR},
        ]
    }
    r = client.post("/ingest/text", json=body)
    assert r.status_code == 201 and r.json()["documents"] == 2 and r.json()["chunks"] == 2
    assert [s["source"] for s in client.get("/sources").json()] == ["lidar", "radar"]
    assert client.post("/ingest/text", json={"documents": []}).status_code == 422


def test_delete_source(client):
    upload(client, "battery.txt", BATTERY)
    assert client.delete("/sources/battery.txt").json() == {"source": "battery.txt", "deleted_chunks": 1}
    assert client.get("/sources").json() == []
    assert client.delete("/sources/battery.txt").status_code == 404


# ---- search ----


def test_every_mode_finds_the_matching_document(client):
    for name, text in [("battery.txt", BATTERY), ("radar.txt", RADAR), ("lidar.txt", LIDAR)]:
        upload(client, name, text)
    for mode in ["dense", "bm25", "hybrid"]:
        for rerank in [True, False]:
            r = client.get("/search", params={"q": "doppler radar fog", "mode": mode, "rerank": rerank})
            assert r.status_code == 200
            assert r.json()["results"][0]["source"] == "radar.txt", (mode, rerank)


def test_without_rerank_scores_are_retrieval_scores(client, encoder):
    upload(client, "battery.txt", BATTERY)
    upload(client, "radar.txt", RADAR)
    body = client.get("/search", params={"q": "radar", "rerank": False}).json()
    assert body["reranked"] is False and encoder.rerank_calls == 0
    for hit in body["results"]:
        assert hit["score"] == hit["retrieval_score"] and hit["rank"] == hit["retrieval_rank"]
    assert body["timings"]["rerank_ms"] == 0


def test_reranker_decides_the_final_order(client, encoder):
    for name, text in [("battery.txt", BATTERY), ("radar.txt", RADAR), ("lidar.txt", LIDAR)]:
        upload(client, name, text)
    # A reranker that always prefers the lidar text, whatever retrieval said.
    encoder.rerank = lambda query, texts: [1.0 if "Lidar" in t else 0.0 for t in texts]
    hits = client.get("/search", params={"q": "radar doppler"}).json()["results"]
    assert hits[0]["source"] == "lidar.txt"
    assert hits[0]["retrieval_rank"] > 1  # it was not first before reranking
    assert hits[0]["score"] == 1.0


def test_search_validation(client):
    assert client.get("/search").status_code == 422  # q is required
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "x", "k": 51}).status_code == 422
    assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422
    assert client.get("/search", params={"q": "x", "offset": 201}).status_code == 422
    assert client.get("/search", params={"q": "x", "mode": "magic"}).status_code == 422
    assert client.get("/search", params={"q": "x"}).json()["results"] == []  # empty collection is fine


# ---- filters and pagination ----


def test_source_filter_accepts_several_sources(client):
    for name, text in [("battery.txt", BATTERY), ("radar.txt", RADAR), ("lidar.txt", LIDAR)]:
        upload(client, name, text)
    only = client.get("/search", params={"q": "radar", "source": "lidar.txt"}).json()["results"]
    assert {h["source"] for h in only} == {"lidar.txt"}
    two = client.get("/search", params={"q": "radar", "source": ["lidar.txt", "battery.txt"]}).json()["results"]
    assert {h["source"] for h in two} == {"lidar.txt", "battery.txt"}


def test_date_filter_is_inclusive(client):
    upload(client, "old.txt", RADAR, date="2026-01-01")
    upload(client, "mid.txt", RADAR, date="2026-06-15")
    upload(client, "new.txt", RADAR, date="2026-09-30")

    def sources(**params):
        return sorted(h["source"] for h in client.get("/search", params={"q": "radar", **params}).json()["results"])

    assert sources(date_from="2026-06-15") == ["mid.txt", "new.txt"]
    assert sources(date_to="2026-06-15") == ["mid.txt", "old.txt"]
    assert sources(date_from="2026-06-01", date_to="2026-06-30") == ["mid.txt"]
    assert client.get("/search", params={"q": "x", "date_from": "2026-02-01", "date_to": "2026-01-01"}).status_code == 422
    hit = client.get("/search", params={"q": "radar", "source": "old.txt"}).json()["results"][0]
    assert hit["date"] == "2026-01-01"


def test_pages_neither_overlap_nor_skip_results(make_client, encoder):
    # Reranker pool of 4 with 12 matching chunks: pages cross the border between reranked and not-reranked results.
    client = make_client(rerank_candidates=4)
    for i in range(12):
        upload(client, f"doc{i:02d}.txt", f"radar note {i} " + "radar " * (i % 5) + "filler " * i)
    for rerank in [True, False]:
        params = {"q": "radar note", "rerank": rerank}
        everything = ids(client.get("/search", params={**params, "k": 12}))
        paged = []
        for offset in range(0, 12, 3):
            page = client.get("/search", params={**params, "k": 3, "offset": offset})
            assert [h["rank"] for h in page.json()["results"]] == [offset + 1, offset + 2, offset + 3]
            paged += ids(page)
        assert paged == everything and len(set(everything)) == 12, rerank


def test_page_past_the_reranker_pool_skips_the_reranker(make_client, encoder):
    client = make_client(rerank_candidates=2)
    for i in range(6):
        upload(client, f"doc{i}.txt", f"radar note {i}")
    client.get("/search", params={"q": "radar", "k": 2, "offset": 4})
    assert encoder.rerank_calls == 0


def test_candidates_parameter_sets_the_reranker_pool(make_client, encoder):
    client = make_client(rerank_candidates=5)
    for i in range(12):
        upload(client, f"doc{i:02d}.txt", f"radar note {i}")
    body = client.get("/search", params={"q": "radar", "k": 3}).json()
    assert body["candidates"] == 5 and encoder.reranked_texts == 5
    body = client.get("/search", params={"q": "radar", "k": 3, "candidates": 10}).json()
    assert body["candidates"] == 10 and encoder.reranked_texts == 15
    assert client.get("/search", params={"q": "radar", "rerank": False}).json()["candidates"] == 0
    assert client.get("/search", params={"q": "radar", "candidates": 101}).status_code == 422
