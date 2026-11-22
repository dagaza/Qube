import unittest
from pathlib import Path

from core.library_ingest_ui_flow import (
    build_conversation_ingest_attach_plan,
    filter_library_ingest_paths,
    filter_paths_after_duplicate_prompt,
    ingestion_file_done_triggers_attach,
    parse_ingestion_file_done_message,
    paths_from_drop_urls,
)


class TestLibraryIngestUiFlow(unittest.TestCase):
    def test_filter_library_ingest_paths(self):
        paths = filter_library_ingest_paths(
            [
                Path("/tmp/a.pdf"),
                Path("/tmp/b.exe"),
                Path("/tmp/c.pdf"),
                Path("/tmp/c.pdf"),
            ]
        )
        self.assertEqual([p.name for p in paths], ["a.pdf", "c.pdf"])

    def test_paths_from_drop_urls_file_uri(self):
        paths = paths_from_drop_urls(["file:///tmp/note.md"])
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].name, "note.md")

    def test_duplicate_decline_filters_paths(self):
        paths = [
            Path("/tmp/a.pdf"),
            Path("/tmp/b.pdf"),
        ]
        kept = filter_paths_after_duplicate_prompt(
            paths,
            {"a.pdf"},
            user_confirmed_overwrite=False,
        )
        self.assertEqual([p.name for p in kept], ["b.pdf"])

    def test_parse_ingestion_file_done(self):
        self.assertEqual(
            parse_ingestion_file_done_message("Indexed: report.pdf"),
            ("indexed", "report.pdf"),
        )
        self.assertEqual(
            parse_ingestion_file_done_message("Skipped (already indexed): old.txt"),
            ("skipped", "old.txt"),
        )

    def test_ingestion_file_done_triggers_attach(self):
        plan = build_conversation_ingest_attach_plan("sess-1", [Path("/tmp/x.pdf")])
        assert plan is not None
        self.assertTrue(
            ingestion_file_done_triggers_attach("Indexed: x.pdf", plan)
        )
        plan.attach_applied = True
        self.assertFalse(
            ingestion_file_done_triggers_attach("Indexed: x.pdf", plan)
        )

    def test_build_conversation_ingest_attach_plan(self):
        plan = build_conversation_ingest_attach_plan(
            "abc",
            [Path("/a/first.pdf"), Path("/a/second.pdf")],
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.attach_basename, "first.pdf")
        self.assertEqual(plan.also_ingested, ["second.pdf"])


if __name__ == "__main__":
    unittest.main()