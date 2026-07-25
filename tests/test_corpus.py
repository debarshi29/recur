from __future__ import annotations

from eval.corpus import search


def test_search_returns_relevant_document():
    result = search("quantum chemistry message passing")
    assert "quantum chemical" in result.lower() or "quantum chemistry" in result.lower()


def test_search_returns_no_match_message_for_unrelated_query():
    result = search("xyzzy plugh nonexistent gibberish query")
    assert result == "No relevant documents found."


def test_search_respects_top_k():
    result = search("graph neural network attention convolution", top_k=1)
    assert result.count("\n\n") == 0
