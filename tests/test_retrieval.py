from __future__ import annotations

from legal_rag.retrieval import deduplicate_parents, rrf_fusion, tokenize


def test_rrf() -> None:
    result = rrf_fusion({
        "dense": [{"idx": 0, "rank": 1, "score": .9}, {"idx": 1, "rank": 2, "score": .8}],
        "bm25": [{"idx": 1, "rank": 1, "score": 5}, {"idx": 0, "rank": 2, "score": 4}],
    })
    assert {x["idx"] for x in result} == {0, 1}
    assert result[0]["score"] == result[1]["score"]


def test_parent_dedup_after_ranking() -> None:
    chunks = [{"parent_section_uid": "a"}, {"parent_section_uid": "a"},
              {"parent_section_uid": "b"}]
    result = deduplicate_parents([{"idx": 1}, {"idx": 0}, {"idx": 2}], chunks, 2)
    assert [r["idx"] for r in result] == [1, 2]


def test_legal_tokenizer() -> None:
    assert "9.2" in tokenize("Section 9.2: change-of-control")
    assert "change-of-control" in tokenize("Section 9.2: change-of-control")
