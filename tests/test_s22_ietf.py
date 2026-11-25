"""Plan S22 (D38): the IETF RFC adapter searches Datatracker's document resource.

It used to send the whole question as ``title__icontains`` to the API index
(``/api/v1/doc/``), which holds no documents, so it returned nothing for every
query (0 of 18 in the live check, 2026-11-25). Datatracker filters by
substring and cannot order results, so the adapter now asks for the most
specific term in the title and the next one in the abstract, falls back to the
title term alone, and ranks locally (13 of 18 non-empty; the 5 empty ones are
outside networking).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.adapters import ietf_rfc  # noqa: E402
from core.knowledge.http_client import BudgetExhaustedError  # noqa: E402

# Public RFC metadata, trimmed.
_TLS13 = {
    "name": "rfc8446", "rfc_number": 8446, "std_level": "/api/v1/name/stdlevelname/ps/",
    "title": "The Transport Layer Security (TLS) Protocol Version 1.3",
    "abstract": "This document specifies version 1.3 of the Transport Layer Security (TLS) protocol.",
    "time": "2026-05-20T15:34:35Z",
}
_TRACES = {
    "name": "rfc8448", "rfc_number": 8448, "std_level": "/api/v1/name/stdlevelname/inf/",
    "title": "Example Handshake Traces for TLS 1.3", "abstract": "Examples of TLS 1.3 handshakes.",
}
_ATLS = {"name": "rfc9999", "rfc_number": 9999, "title": "ATLS over CoAP", "abstract": "Version 1.3 of ATLS."}
_OTHER_TLS = {"name": "rfc5246", "rfc_number": 5246, "title": "The TLS Protocol Version 1.2", "abstract": "TLS."}


def _response(objects):
    resp = MagicMock()
    resp.json.return_value = {"objects": objects}
    resp.raise_for_status.return_value = None
    return resp


class TermTests(unittest.TestCase):
    def test_question_words_and_ietf_words_are_dropped(self) -> None:
        self.assertEqual(ietf_rfc.query_terms("How does TLS 1.3 handshake work?"), ["TLS", "1.3", "handshake"])
        self.assertEqual(ietf_rfc.query_terms("Which RFC defines the HTTP/3 protocol"), ["HTTP/3"])
        self.assertEqual(ietf_rfc.query_terms("what is the"), [])

    def test_the_most_specific_term_goes_in_the_title(self) -> None:
        cases = {
            "How does TLS 1.3 handshake work": ("TLS", "1.3"),
            "OAuth 2.0 PKCE": ("OAuth", "PKCE"),
            "HTTP/2 server push": ("HTTP/2", "server"),
            "IPv6 address autoconfiguration": ("IPv6", "autoconfiguration"),
            "Which RFC defines the HTTP/3 protocol": ("HTTP/3", None),
            "Kerberos authentication": ("authentication", "Kerberos"),
            "what is the": (None, None),
        }
        for query, want in cases.items():
            with self.subTest(query=query):
                self.assertEqual(ietf_rfc.search_terms(ietf_rfc.query_terms(query)), want)


class FetchTests(unittest.TestCase):
    def test_title_and_abstract_terms_on_the_document_resource(self) -> None:
        with patch.object(ietf_rfc, "knowledge_get", return_value=_response([_TLS13])) as get:
            payload = ietf_rfc.fetch_search_results("How does TLS 1.3 handshake work")
        self.assertEqual(payload, {"objects": [_TLS13]})
        self.assertEqual(get.call_count, 1)
        url = get.call_args.args[0]
        params = get.call_args.kwargs["params"]
        self.assertEqual(url, "https://datatracker.ietf.org/api/v1/doc/document/")
        self.assertEqual(
            params,
            {"title__icontains": "TLS", "abstract__icontains": "1.3", "type": "rfc",
             "limit": ietf_rfc.FETCH_LIMIT, "format": "json"},
        )

    def test_title_term_alone_when_the_pair_finds_nothing(self) -> None:
        with patch.object(ietf_rfc, "knowledge_get", side_effect=[_response([]), _response([_TLS13])]) as get:
            payload = ietf_rfc.fetch_search_results("TLS frobnication")
        self.assertEqual(payload["objects"], [_TLS13])
        self.assertNotIn("abstract__icontains", get.call_args_list[1].kwargs["params"])

    def test_no_content_terms_means_no_request(self) -> None:
        with patch.object(ietf_rfc, "knowledge_get") as get:
            self.assertEqual(ietf_rfc.search_ietf_rfc("what is the"), [])
            self.assertEqual(ietf_rfc.fetch_search_results("what is the"), {"objects": []})
        get.assert_not_called()

    def test_budget_exhaustion_returns_nothing(self) -> None:
        err = BudgetExhaustedError(host="datatracker.ietf.org", metrics_host="datatracker.ietf.org")
        with patch.object(ietf_rfc, "knowledge_get", side_effect=err):
            self.assertEqual(ietf_rfc.search_ietf_rfc("TLS 1.3"), [])


class RankTests(unittest.TestCase):
    def _search(self, query, objects, n=3):
        with patch.object(ietf_rfc, "fetch_search_results", return_value={"objects": objects}):
            return [r["doc_name"] for r in ietf_rfc.search_ietf_rfc(query, max_results=n)]

    def test_whole_words_rank_title_matches_first(self) -> None:
        got = self._search("How does TLS 1.3 handshake work", [_ATLS, _OTHER_TLS, _TLS13, _TRACES])
        # ATLS is not TLS; RFC 5246 matches one term of three.
        self.assertEqual(got, ["rfc8448", "rfc8446"])

    def test_ties_go_to_the_shorter_title_then_the_newer_rfc(self) -> None:
        a = {"name": "rfc1000", "rfc_number": 1000, "title": "QUIC Transport"}
        b = {"name": "rfc2000", "rfc_number": 2000, "title": "QUIC Transport"}
        c = {"name": "rfc3000", "rfc_number": 3000, "title": "QUIC Transport Extensions"}
        self.assertEqual(self._search("QUIC transport", [a, c, b]), ["rfc2000", "rfc1000", "rfc3000"])
        self.assertEqual(self._search("QUIC transport", [a, c, b], n=1), ["rfc2000"])

    def test_one_term_query_needs_one_match(self) -> None:
        h3 = {"name": "rfc9114", "rfc_number": 9114, "title": "HTTP/3"}
        h2 = {"name": "rfc9113", "rfc_number": 9113, "title": "HTTP/2"}
        self.assertEqual(self._search("Which RFC defines the HTTP/3 protocol", [h2, h3]), ["rfc9114"])


class RowTests(unittest.TestCase):
    def test_row_carries_level_and_abstract_but_no_date(self) -> None:
        row = ietf_rfc._row_from_doc(_TLS13)
        self.assertEqual(row["title"], "RFC8446 — The Transport Layer Security (TLS) Protocol Version 1.3")
        self.assertTrue(row["snippet"].startswith("RFC8446 (Proposed Standard): The Transport Layer"))
        self.assertIn("specifies version 1.3", row["snippet"])
        self.assertIn("specifies version 1.3", row["full_text"])
        self.assertEqual(row["url"], "https://www.rfc-editor.org/rfc/rfc8446.html")
        # Datatracker's ``time`` is the record's last change, not the publication.
        self.assertIsNone(row["publication_date"])
        self.assertEqual(row["document_type"], "standard_document")

    def test_unknown_level_and_missing_abstract(self) -> None:
        row = ietf_rfc._row_from_doc({"name": "rfc1", "title": "Host Software", "std_level": None})
        self.assertEqual(row["snippet"], "RFC1: Host Software")
        self.assertEqual(row["full_text"], "Host Software")


if __name__ == "__main__":
    unittest.main()
