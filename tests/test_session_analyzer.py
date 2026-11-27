"""Offline regression coverage; execute with python -S -m unittest discover ..."""
from __future__ import annotations

from contextlib import closing, redirect_stdout, redirect_stderr
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from scripts.analyze_session import discover, main
from session_analyzer.analysis import analyze, image_file
from session_analyzer.reader import read_session, snapshot
from session_analyzer.report import safe_name, write_report
from session_analyzer.source import ROOT, code_defaults, literals
from session_analyzer.synthetic import generate


def digest_tree(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="clue-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session = generate(self.root / "한글 경로 # 세션% &")

    def data(self):
        return analyze(read_session(self.session))

    def edit(self, sql):
        with closing(sqlite3.connect(self.session / "metadata.db")) as conn, conn:
            conn.executescript(sql)

    def test_schema_is_current_without_runtime_import(self):
        with closing(sqlite3.connect(self.session / "metadata.db")) as conn:
            schema = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(schema, set(literals("context_qa/store.py")["TABLES"]))
        # Isolate module state from other CLUE tests which may import AI packages.
        code = ("import sys; from pathlib import Path; from session_analyzer.reader import read_session; "
                "from session_analyzer.analysis import analyze; analyze(read_session(Path(sys.argv[1]))); "
                "assert not any(n in sys.modules for n in ('torch','paddle','numpy','faster_whisper'))")
        result = subprocess.run([sys.executable, "-S", "-c", code, str(self.session)], cwd=ROOT, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_metrics_match_direct_sql_and_exclude_snapshots(self):
        data = self.data()
        with closing(sqlite3.connect(self.session / "metadata.db")) as conn:
            observations = conn.execute("SELECT count(*) FROM frame_history WHERE source='periodic'").fetchone()[0]
            frames = conn.execute("SELECT count(*) FROM frames WHERE source='periodic'").fetchone()[0]
            completed = conn.execute("SELECT elapsed_ms FROM answers WHERE status='completed' ORDER BY elapsed_ms").fetchall()
        s = data["summary"]
        self.assertEqual(s["periodic_observations"], observations)
        self.assertEqual(s["periodic_frames"], frames)
        self.assertAlmostEqual(s["storage_reduction_pct"], (1 - frames / observations) * 100)
        self.assertEqual(s["completed_total_ms"]["median"], completed[len(completed)//2][0])
        self.assertEqual(sum(s["classification"].values()), frames)
        self.assertEqual(s["snapshot_start"], 8)
        self.assertEqual(s["snapshot_submit"], 8)

    def test_failed_cancelled_unanswered_excluded_from_performance(self):
        self.edit("UPDATE answers SET elapsed_ms=9999999 WHERE status != 'completed'")
        s = self.data()["summary"]
        self.assertEqual(s["completed_total_ms"]["count"], 5)
        self.assertLess(s["completed_total_ms"]["max"], 9999999)
        self.assertEqual(s["statuses"], {"completed": 5, "failed": 1, "cancelled": 1, "unanswered": 1})

    def test_no_questions(self):
        self.edit("DELETE FROM answers; DELETE FROM questions;")
        data = self.data()
        self.assertEqual(data["questions"], [])
        self.assertIsNone(data["summary"]["completed_total_ms"]["median"])
        write_report(data, self.session, self.root / "reports")

    def test_no_transcripts(self):
        self.edit("DELETE FROM transcripts;")
        data = self.data()
        self.assertEqual(data["summary"]["transcript_count"], 0)
        self.assertIsNone(data["summary"]["transcript_range"]["start_ms"])
        self.assertTrue(any("参照" in w or "참조" in w for w in data["warnings"]))

    def test_no_frames(self):
        self.edit("DELETE FROM frames; DELETE FROM frame_history;")
        data = self.data()
        self.assertEqual(data["summary"]["periodic_frames"], 0)
        self.assertIsNone(data["summary"]["storage_reduction_pct"])
        self.assertTrue(all(t["observed_frame_id"] is None for t in data["transcripts"]))

    def test_no_rag_chunks(self):
        self.edit("DELETE FROM text_chunks; DELETE FROM chunk_embeddings;")
        self.assertEqual(self.data()["chunks"], [])

    def test_missing_images_preserves_frames_and_selected_ids(self):
        path = next((self.session / "frames").glob("*.svg"))
        path.unlink()
        data = self.data()
        self.assertEqual(data["summary"]["missing_images"], 1)
        self.assertEqual(data["summary"]["frame_count"], 31)
        self.assertTrue(data["questions"][0]["used_frame_ids"])
        write_report(data, self.session, self.root / "reports")

    def test_no_image_files(self):
        shutil.rmtree(self.session / "frames")
        data = self.data()
        self.assertEqual(data["summary"]["missing_images"], 31)
        self.assertEqual(data["summary"]["image_bytes"], 0)

    def test_legacy_schema_missing_columns_does_not_infer_completed(self):
        self.edit("ALTER TABLE answers DROP COLUMN status; ALTER TABLE questions DROP COLUMN question_type; ALTER TABLE answers DROP COLUMN input_tokens;")
        data = self.data()
        self.assertEqual(data["summary"]["statuses"], {"unknown": 7, "unanswered": 1})
        self.assertIsNone(data["summary"]["completed_total_ms"]["median"])
        self.assertTrue(data["warnings"])

    def test_missing_table_is_not_zero(self):
        self.edit("DROP TABLE frame_history; DROP TABLE chunk_embeddings; DROP TABLE transcripts;")
        data = self.data()
        self.assertIsNone(data["summary"]["periodic_observations"])
        self.assertIsNone(data["summary"]["transcript_count"])
        self.assertIsNone(data["chunks"][0]["embedding_exists"])

    def test_empty_db(self):
        empty = self.root / "빈 DB"
        empty.mkdir()
        sqlite3.connect(empty / "metadata.db").close()
        data = analyze(read_session(empty))
        self.assertIsNone(data["summary"]["question_count"])
        self.assertEqual(data["summary"]["observed_range"], {"start_ms": None, "end_ms": None, "span_ms": None})

    def test_corrupt_db_error(self):
        (self.session / "metadata.db").write_bytes(b"not sqlite")
        with self.assertRaises(sqlite3.DatabaseError):
            read_session(self.session)

    def test_no_sessions_cli(self):
        output = StringIO()
        with redirect_stdout(output):
            result = main(["--sessions-root", str(self.root / "absent"), "--no-open"])
        self.assertEqual(result, 0)
        self.assertIn("분석할 세션이 없습니다", output.getvalue())

    def test_read_and_report_leave_source_byte_identical(self):
        before = digest_tree(self.session)
        write_report(self.data(), self.session, self.root / "reports")
        self.assertEqual(before, digest_tree(self.session))

    def test_uncheckpointed_wal_read_without_new_original_sidecars(self):
        db = self.session / "metadata.db"
        conn = sqlite3.connect(db)
        self.addCleanup(conn.close)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("UPDATE transcripts SET text='WAL에서만 존재하는 발화' WHERE segment_id='s00000'")
        conn.commit()
        before = digest_tree(self.session)
        data = self.data()
        self.assertEqual(data["transcripts"][0]["text"], "WAL에서만 존재하는 발화")
        self.assertEqual(before, digest_tree(self.session))

    def test_snapshot_connection_is_read_only(self):
        with snapshot(self.session / "metadata.db") as conn:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM transcripts")

    def test_active_rollback_journal_rejected(self):
        (self.session / "metadata.db-journal").write_bytes(b"active")
        with self.assertRaises(ValueError):
            read_session(self.session)

    def test_reused_observation_is_used_for_audio_alignment(self):
        self.edit("UPDATE transcripts SET start_ms=16000, end_ms=17000 WHERE segment_id='s00000'")
        segment = next(s for s in self.data()["transcripts"] if s["segment_id"] == "s00000")
        self.assertEqual(segment["observed_frame_id"], "f000000")
        self.assertEqual(segment["observation_ms"], 16000)
        self.assertEqual(segment["observation_decision"], "reused")

    def test_no_future_frame_for_earlier_audio(self):
        self.edit("UPDATE transcripts SET start_ms=0,end_ms=500 WHERE segment_id='s00000'")
        self.assertIsNone(self.data()["transcripts"][0]["observed_frame_id"])

    def test_overlap_gaps_use_union(self):
        self.edit("DELETE FROM transcripts; INSERT INTO transcripts VALUES ('SYNTHETIC_멘토링_데모','a',0,10000,'a'),('SYNTHETIC_멘토링_데모','b',2000,3000,'b'),('SYNTHETIC_멘토링_데모','c',12000,13000,'c')")
        self.assertEqual(self.data()["gaps"], [{"start_ms":10000,"end_ms":12000,"duration_ms":2000}])

    def test_recorded_context_order_and_null_rag_preserved(self):
        self.edit("UPDATE answers SET used_frame_ids='[\"q001_question_submit\",\"f000000\",\"q001_question_start\"]' WHERE question_id='q001'")
        q = self.data()["questions"][0]
        self.assertEqual(q["used_frame_ids"], ["q001_question_submit","f000000","q001_question_start"])
        self.assertIsNone(q["rag_performed"])
        self.assertTrue(self.data()["questions"][2]["rag_performed"])

    def test_corrupt_ids_are_unknown_not_empty(self):
        self.edit("UPDATE answers SET used_frame_ids='broken',used_segment_ids='[1]' WHERE question_id='q001'")
        data = self.data()
        self.assertIsNone(data["questions"][0]["used_frame_ids"])
        self.assertIsNone(data["questions"][0]["used_segment_ids"])

    def test_orphan_answers_are_visible(self):
        self.edit("DELETE FROM questions WHERE question_id='q001'")
        data = self.data()
        self.assertEqual(data["summary"]["orphan_answer_count"], 1)
        self.assertEqual(len(data["questions"]), 8)

    def test_multiple_sessions_require_explicit_selection(self):
        self.edit("INSERT INTO transcripts VALUES ('other','other-s',0,1,'다른 세션')")
        with self.assertRaises(ValueError):
            read_session(self.session)
        data = analyze(read_session(self.session,"other"))
        self.assertEqual(len(data["transcripts"]), 1)
        self.assertEqual(data["frames"], [])

    def test_path_traversal_and_external_image_denied(self):
        outside = self.root / "secret.jpg"
        outside.write_bytes(b"private")
        self.assertIsNone(image_file(self.session, "../secret.jpg"))
        self.assertIsNone(image_file(self.session, str(outside)))
        self.assertIsNone(image_file(self.session, "..\\secret.jpg"))

    def test_jpeg_storage_counts_real_files_not_db_rows(self):
        image = self.session / "frames" / "synthetic.jpg"
        image.write_bytes(b"\xff\xd8synthetic-jpeg-marker\xff\xd9")
        self.edit("UPDATE frames SET image_path='frames/synthetic.jpg' WHERE frame_id='f000000'")
        data = self.data()
        self.assertEqual(data["summary"]["jpeg_count"], 1)
        self.assertEqual(data["summary"]["jpeg_bytes"], image.stat().st_size)

    def test_jpeg_storage_includes_unreferenced_files_once(self):
        image = self.session / "frames" / "orphan.jpg"
        image.write_bytes(b"\xff\xd8orphan\xff\xd9")
        self.edit("UPDATE frames SET image_path='frames/orphan.jpg' WHERE frame_id IN ('f000000','f000012')")
        unreferenced = self.session / "frames" / "unreferenced.jpeg"
        unreferenced.write_bytes(b"\xff\xd8extra\xff\xd9")
        data = self.data()
        self.assertEqual(data["summary"]["jpeg_count"], 2)
        self.assertEqual(data["summary"]["jpeg_bytes"], image.stat().st_size + unreferenced.stat().st_size)

    def test_safe_export_html_and_spreadsheet_formulas(self):
        self.edit("UPDATE questions SET text='</script><script>alert(1)</script>' WHERE question_id='q001'; UPDATE transcripts SET text='=HYPERLINK(\"https://example.invalid\")' WHERE segment_id='s00000'")
        report = write_report(self.data(), self.session, self.root / "reports")
        page = (report / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("</script><script>alert", page)
        self.assertIn("\\u003c/script", page)
        with (report / "transcript_analysis.csv").open(encoding="utf-8-sig",newline="") as stream:
            record = next(csv.DictReader(stream))
        self.assertTrue(record["text"].startswith("'="))
        self.assertEqual(record["synthetic"], "True")
        self.assertIn("합성 데이터", record["data_origin"])

    def test_common_secrets_masked_and_absolute_paths_not_exported(self):
        self.edit("UPDATE transcripts SET text='API_KEY=abc123456789abcdef person@example.com 010-1234-5678 sk-proj-abcdefghijklmnopqrstuv' WHERE segment_id='s00000'")
        report = write_report(self.data(), self.session, self.root / "reports")
        text = (report / "session_analysis.json").read_text(encoding="utf-8")
        self.assertNotIn("abc123456789abcdef", text)
        self.assertNotIn("person@example.com", text)
        self.assertNotIn("010-1234-5678", text)
        self.assertNotIn("sk-proj-abcdefghijklmnopqrstuv", text)
        self.assertNotIn(str(self.session), text)

    def test_report_is_portable_and_repeated_runs_do_not_overwrite(self):
        a = write_report(self.data(), self.session, self.root / "reports")
        b = write_report(self.data(), self.session, self.root / "reports")
        self.assertNotEqual(a,b)
        moved = self.root / "옮긴 보고서"
        shutil.copytree(a,moved)
        data = json.loads((moved / "session_analysis.json").read_text(encoding="utf-8"))
        for frame in data["frames"]:
            self.assertTrue((moved / frame["asset"]).is_file())
        for file in ["summary.json","frame_analysis.csv","frame_metadata.csv","transcript_analysis.csv","rag_chunks.csv","question_context.csv","response_analysis.csv","dashboard.css","dashboard.js","index.html"]:
            self.assertTrue((moved / file).is_file())

    def test_refuse_output_inside_source(self):
        with self.assertRaises(ValueError):
            write_report(self.data(), self.session, self.session / "analysis")

    def test_cli_explicit_and_latest_without_browser(self):
        with redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            self.assertEqual(main([str(self.session),"--output-root",str(self.root/"report-a"),"--no-open"]),0)
            self.assertEqual(main(["--sessions-root",str(self.root),"--latest","--output-root",str(self.root/"report-b"),"--no-open"]),0)
        self.assertEqual(discover(self.root)[0]["folder"], self.session)

    def test_source_defaults_match_checked_out_code(self):
        settings=code_defaults()
        self.assertEqual(settings["pc_threshold"],0.90)
        self.assertEqual(settings["rc_threshold"],0.98)
        self.assertEqual(settings["audio_period_seconds"],5.0)
        self.assertEqual(settings["audio_overlap_seconds"],1.0)
        self.assertEqual(settings["boundary_holdback_ms"],250)

    def test_generator_does_not_overwrite(self):
        before=digest_tree(self.session)
        with self.assertRaises(FileExistsError):
            generate(self.session)
        self.assertEqual(before,digest_tree(self.session))

    def test_report_name_cannot_escape_or_use_reserved_windows_name(self):
        for value in ("../../escape", "CON", "", "a/b", "a?b"):
            name=safe_name(value)
            self.assertNotIn("/",name)
            self.assertNotIn("\\",name)
            self.assertFalse(name.startswith("."))
        self.assertNotEqual(safe_name("a/b"),safe_name("a?b"))

    def test_large_session_uses_bounded_assets_and_all_exports(self):
        folder=generate(self.root/"장시간 세션",observations=10080)  # 14 hours, 840 representative frames
        data=analyze(read_session(folder))
        self.assertEqual(data["summary"]["periodic_observations"],10080)
        self.assertEqual(data["summary"]["periodic_frames"],840)
        self.assertEqual(len(data["history"]),10096)
        report=write_report(data,folder,self.root/"large-reports")
        with (report/"frame_analysis.csv").open(encoding="utf-8-sig",newline="") as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)),10096)

    def test_cli_runs_with_site_packages_disabled(self):
        result=subprocess.run([sys.executable,"-S","-m","scripts.analyze_session",str(self.session),"--no-open","--output-root",str(self.root/"stdlib-only")],cwd=ROOT,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=="__main__":
    unittest.main()
