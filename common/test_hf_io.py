"""Offline tests for common/hf_io.py — hub calls are faked, parquet is real.

The ``huggingface_hub`` calls (list/download) are monkeypatched so no network
is touched; text extraction runs on a tiny locally-written parquet so the
reader path is exercised for real.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def hf(monkeypatch, tmp_path):
    import common.hf_io as h

    monkeypatch.setattr(h, "hf_available", lambda: True)
    return h, tmp_path


def test_hf_available_false_when_not_installed(monkeypatch):
    import common.hf_io as h

    monkeypatch.setattr(h, "hf_available", lambda: False)
    assert h.hf_available() is False
    with pytest.raises(h.HFError, match="huggingface_hub not installed"):
        h.list_repo_files("owner/repo")


def test_list_repo_files_prefix_filter(hf, monkeypatch):
    h, _ = hf

    def fake_list(repo_id, repo_type="dataset", token=None):
        assert repo_id == "owner/repo"
        assert repo_type == "dataset"
        return ["a/one.parquet", "a/two.parquet", "b/three.parquet"]

    monkeypatch.setattr("common.hf_io.hf_available", lambda: True)
    monkeypatch.setattr("huggingface_hub.list_repo_files", fake_list)
    assert h.list_repo_files("owner/repo", prefix="a/") == [
        "a/one.parquet",
        "a/two.parquet",
    ]


def test_list_repo_files_gated_hint(hf, monkeypatch):
    h, _ = hf

    def boom(**kw):
        raise RuntimeError("Repository is gated. ... 401 Unauthorized")

    monkeypatch.setattr("huggingface_hub.list_repo_files", boom)
    with pytest.raises(h.HFError, match="gated"):
        h.list_repo_files("owner/gated")


def test_hf_download_returns_cached_path(hf, monkeypatch, tmp_path):
    h, _ = hf
    calls = {}

    def fake_download(repo_id, filename, cache_dir, repo_type="dataset", token=None, etag_timeout=120):
        calls["args"] = (repo_id, filename, cache_dir, repo_type)
        p = Path(tmp_path) / "downloaded.parquet"
        p.write_text("x", encoding="utf-8")
        return str(p)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    out = h.hf_download("owner/repo", "data/x.parquet", str(tmp_path / "cache"))
    assert out.name == "downloaded.parquet"
    assert calls["args"][0] == "owner/repo"
    assert calls["args"][1] == "data/x.parquet"
    assert calls["args"][3] == "dataset"


def test_iter_parquet_text_picks_text_column(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from common.hf_io import iter_parquet_text

    table = pa.table(
        {
            "id": [1, 2],
            "text": ["भारत एक देश है", "नई दिल्ली राजधानी है"],
        }
    )
    path = tmp_path / "docs.parquet"
    pq.write_table(table, str(path))
    docs = list(iter_parquet_text([str(path)]))
    assert len(docs) == 2
    assert docs[0][0] == "docs.parquet#0"
    assert "भारत" in docs[0][1]


def test_iter_parquet_text_falls_back_to_string_column(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from common.hf_io import iter_parquet_text

    table = pa.table({"title": ["a"], "body": ["some body text"]})
    path = tmp_path / "x.parquet"
    pq.write_table(table, str(path))
    docs = list(iter_parquet_text([str(path)]))
    assert docs[0][1] == "some body text"  # 'body' is in TEXT_KEYS


def test_iter_parquet_text_no_text_column_raises(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from common.hf_io import HFError, iter_parquet_text

    path = tmp_path / "n.parquet"
    pq.write_table(pa.table({"n": [1, 2]}), str(path))
    with pytest.raises(HFError, match="no text column"):
        list(iter_parquet_text([str(path)]))


def test_iter_text_lines(tmp_path):
    from common.hf_io import iter_text_lines

    p = tmp_path / "corpus.txt"
    p.write_text("पहली पंक्ति\n\nदूसरी पंक्ति\n", encoding="utf-8")
    docs = list(iter_text_lines([str(p)]))
    assert [d[1] for d in docs] == ["पहली पंक्ति", "दूसरी पंक्ति"]
    assert docs[0][0] == "corpus.txt#0"


def test_iter_jsonl_text(tmp_path):
    from common.hf_io import iter_jsonl_text

    p = tmp_path / "lines.jsonl"
    p.write_text(
        json.dumps({"text": "कुछ पाठ"}, ensure_ascii=False)
        + "\n"
        + "not json\n"
        + json.dumps({"sentence": "दूसरा पाठ"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    docs = list(iter_jsonl_text([str(p)]))
    assert [d[1] for d in docs] == ["कुछ पाठ", "दूसरा पाठ"]


def test_iter_csv_text(tmp_path):
    from common.hf_io import iter_csv_text

    p = tmp_path / "docs.csv"
    p.write_text('title,text\n"एक","पहला दस्तावेज़"\n"दो","दूसरा दस्तावेज़"\n', encoding="utf-8")
    docs = list(iter_csv_text([str(p)]))
    assert [d[1] for d in docs] == ["पहला दस्तावेज़", "दूसरा दस्तावेज़"]


def test_hf_token_from_env(monkeypatch):
    import common.hf_io as h

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    assert h.hf_token() is None
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert h.hf_token() == "hf_x"
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_y")
    assert h.hf_token() == "hf_y"
