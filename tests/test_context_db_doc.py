from __future__ import annotations

import unittest

from context_qa.store import TABLES
from scripts.gen_db_doc import COLUMN_NOTES, DOC_PATH, TABLE_NOTES, render


class DBDocTests(unittest.TestCase):
    def test_doc_matches_schema(self) -> None:
        self.assertEqual(
            DOC_PATH.read_text(encoding="utf-8"),
            render(),
            "run: python -m scripts.gen_db_doc",
        )

    def test_every_table_and_column_described(self) -> None:
        import sqlite3

        from context_qa.store import SCHEMA

        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        self.assertEqual(set(TABLE_NOTES), set(TABLES))
        for table in TABLES:
            for row in conn.execute(f"PRAGMA table_info({table})"):
                name = row[1]
                if name != "session_id":
                    self.assertIn((table, name), COLUMN_NOTES, f"{table}.{name}")
        conn.close()


if __name__ == "__main__":
    unittest.main()
