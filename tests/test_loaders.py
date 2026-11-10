from app.ingestion import loaders
from app.ingestion.loaders import LoadedDocument, chunk_text


def test_chunking_is_deterministic_and_overlaps():
    text = "alpha beta gamma delta epsilon " * 20
    first = chunk_text(text, size=80, overlap=20)
    assert first == chunk_text(text, size=80, overlap=20)
    assert len(first) > 1
    assert first[0].split()[-2] in first[1]


def test_chunking_handles_unicode_whitespace_and_punctuation():
    chunks = chunk_text("  Café — déjà vu!\n\n第二段？  " * 5, size=45, overlap=8)
    assert chunks
    assert all(chunk.strip() == chunk for chunk in chunks)


def test_chunking_rejects_invalid_overlap():
    try:
        chunk_text("hello", size=20, overlap=20)
    except ValueError as exc:
        assert "greater" in str(exc)
    else:
        raise AssertionError("expected invalid chunk configuration to fail")


def test_limit_truncates_location_segments_used_by_the_worker(monkeypatch):
    monkeypatch.setattr(loaders.settings, "max_document_chars", 7)
    loaded = LoadedDocument(
        "abcd\nefgh\nijkl",
        {},
        (("abcd", {"page": 1}), ("efgh", {"page": 2}), ("ijkl", {"page": 3})),
    )

    limited = loaders._limit(loaded)

    assert limited.text == "abcd\nef"
    assert limited.segments == (("abcd", {"page": 1}), ("efg", {"page": 2}))
    assert sum(len(text) for text, _ in limited.segments) <= 7
