"""Test Q&A-weighted transcript truncation (#21).

A plain head cut keeps the scripted prepared remarks and drops the analyst Q&A;
this verifies the weighted cut keeps the Q&A (where the signal is) while still
retaining a prepared-remarks head, within the same character budget.
"""
import re, os

_src = open(os.path.join(os.path.dirname(__file__), "..", "app", "ml", "llm_scoring.py")).read()
_block = re.search(r"import re as _re.*?return prepared\[:prep_take\] \+ sep \+ qa\[:qa_take\]", _src, re.S).group(0)
_ns = {}
exec(_block, _ns)
qa_trunc = _ns["_qa_weighted_truncate"]


def test_qa_kept_prepared_head_kept_within_budget():
    prepared = "PREPARED REMARKS. " * 1200
    qa = "Operator: question-and-answer session. " + ("ANALYST_QA_SIGNAL " * 1200)
    text = prepared + qa
    limit = 15_000
    naive = text[:limit]
    weighted = qa_trunc(text, limit)
    assert "ANALYST_QA_SIGNAL" not in naive          # naive drops Q&A
    assert "ANALYST_QA_SIGNAL" in weighted            # weighted keeps it
    assert "PREPARED REMARKS" in weighted             # and a prepared head
    assert len(weighted) <= limit


def test_short_returned_whole():
    s = "Operator: questions. short text"
    assert qa_trunc(s, 15_000) == s


def test_no_marker_falls_back_to_head_cut():
    nm = "JUST PREPARED. " * 2000
    assert qa_trunc(nm, 15_000) == nm[:15_000]


def test_empty_safe():
    assert qa_trunc("", 15_000) == "" and qa_trunc(None, 15_000) == ""
