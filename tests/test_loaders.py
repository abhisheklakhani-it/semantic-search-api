from pathlib import Path

import pytest

from semsearch.loaders import UnsupportedFileType, load_bytes, load_file, split_documents
from tests.pdf_utils import make_pdf

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_text_file_is_one_document_without_page():
    docs = load_bytes(b"Hello world", "notes.txt")
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "notes.txt"
    assert "page" not in docs[0].metadata


def test_pdf_is_loaded_page_by_page_with_1_based_page_numbers():
    pdf = make_pdf(["Alpha page text.", "Bravo page text.", "Charlie page text."])
    chunks = split_documents(load_bytes(pdf, "report.pdf"), chunk_size=500, chunk_overlap=0)
    assert [(c.page, c.text) for c in chunks] == [
        (1, "Alpha page text."),
        (2, "Bravo page text."),
        (3, "Charlie page text."),
    ]


def test_source_is_the_file_name_without_directories():
    docs = load_bytes(b"text", "../../etc/secret.md")
    assert docs[0].metadata["source"] == "secret.md"


@pytest.mark.parametrize("name", ["data.csv", "image.png", "no_extension"])
def test_unsupported_types_are_rejected(name):
    with pytest.raises(UnsupportedFileType):
        load_bytes(b"x", name)


def test_chunks_respect_size_overlap_and_are_numbered():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = split_documents(load_bytes(text.encode(), "long.txt"), chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Overlap: each chunk starts with words that ended the previous one.
    first_word_of_second = chunks[1].text.split()[0]
    assert first_word_of_second in chunks[0].text


def test_whitespace_from_pdf_line_breaks_is_collapsed():
    chunks = split_documents(load_bytes(b"line one\nline   two\t\tend", "a.txt"), 500, 0)
    assert chunks[0].text == "line one line two end"


def test_example_files_load():
    for path in EXAMPLES.iterdir():
        chunks = split_documents(load_file(path), 800, 100)
        assert chunks, path.name
    pdf_chunks = split_documents(load_file(EXAMPLES / "iso26262.pdf"), 800, 100)
    assert {c.page for c in pdf_chunks} == {1, 2, 3}
