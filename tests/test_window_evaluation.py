import unittest

from interface_spec.contracts import FrameDecision

from frame_processing.experiments.preservation import Event
from frame_processing.experiments.replay import Prediction
from frame_processing.experiments.window_evaluation import (
    STORED, TARGET, UPDATE, VerificationWindow, parse_verification_windows, score_required_frames,
    score_windows, transition_cases,
)


def predictions(first, last, saved):
    return [Prediction(index, (index - 1) * 1000, "0.9", None, None,
                       saved.get(index, FrameDecision.REUSED), "group", "rep", index, index in saved)
            for index in range(first, last + 1)]


class WindowEvaluationTests(unittest.TestCase):
    def test_nonconsecutive_choices_and_new_group_substitution(self):
        events = [Event("event", 7, 12, "required information")]
        checks = [
            VerificationWindow("or", UPDATE, 9, 11, "event", (9, 11)),
            VerificationWindow("exact_hit", UPDATE, 10, 10, "event"),
            VerificationWindow("exact_missed", UPDATE, 11, 11, "event"),
            VerificationWindow("target", TARGET, 9, 11, "event", target_count=2),
            VerificationWindow("representative", STORED, 9, 9, "event"),
        ]
        rows = predictions(7, 12, {9: FrameDecision.NEW_GROUP, 10: FrameDecision.NEW_REPRESENTATIVE})
        summary, details = score_windows("0.9", rows, events, checks)
        self.assertEqual((summary["update_hits"], summary["update_checks"]), (1, 3))
        self.assertEqual(summary["new_group_instead_frames"], "9")
        self.assertEqual(summary["stored_representative_hits"], 1)
        self.assertEqual(summary["reference_targets_met"], 0)
        self.assertEqual([row["status"] for row in details], [
            "hit", "new_group_only", "hit", "missed", "below_reference_target", "hit",
        ])
        self.assertEqual(details[1]["candidate_frames"], "9 11")
        self.assertEqual(details[1]["new_representative_frames"], "")
        self.assertEqual(details[1]["new_group_frames"], "9")

    def test_adjacent_required_windows_and_unwanted_storage_are_separate(self):
        events = [Event("one", 7, 8, "first stage"), Event("two", 9, 10, "second stage")]
        checks = [VerificationWindow("scroll", "scroll", 11, 11, "")]
        rows = predictions(7, 12, {8: FrameDecision.NEW_GROUP, 11: FrameDecision.NEW_GROUP,
                                   12: FrameDecision.NEW_REPRESENTATIVE})
        summary, _ = score_windows("0.9", rows, events, checks)
        self.assertEqual((summary["observed"], summary["preserved_windows"], summary["required_windows"]), (6, 1, 2))
        self.assertEqual(summary["missed_required_windows"], "9-10")
        self.assertEqual((summary["stored_frames"], summary["deduplication_rate"]), (3, 0.5))
        self.assertEqual((summary["unnecessary_stored_frames"], summary["unjudged_stored_frames"]), (1, 1))

    def test_existing_verification_schema_and_candidate_validation(self):
        events = [Event("event", 7, 12, "required information")]
        header = b"check_id,kind,first_frame,last_frame,event_id,description"
        legacy = header + b"\nold,new_representative,8,10,event,update inside range\n"
        self.assertEqual(parse_verification_windows(legacy, events, 7, 12),
                         [VerificationWindow("old", UPDATE, 8, 10, "event")])
        invalid = header + b",candidate_frames\nbad,new_representative,8,10,event,outside,8 11\n"
        with self.assertRaisesRegex(ValueError, "invalid candidate"):
            parse_verification_windows(invalid, events, 7, 12)

    def test_required_points_allow_predecessor_once_without_guessing_negatives(self):
        rows = predictions(7, 12, {7: FrameDecision.NEW_GROUP, 9: FrameDecision.NEW_REPRESENTATIVE,
                                   12: FrameDecision.NEW_GROUP})
        summary, details = score_required_frames("0.9", rows, [8, 9, 10])
        self.assertEqual([row["stored_frames"] for row in details], [7, 9, None])
        self.assertEqual((summary["preserved_windows"], summary["required_windows"]), (2, 3))
        self.assertEqual(summary["missed_required_windows"], "10")
        self.assertIsNone(summary["unnecessary_stored_frames"])
        self.assertEqual((summary["new_groups"], summary["new_representatives"]), (2, 1))

    def test_transition_reviews_distinguish_manual_and_single_threshold_evidence(self):
        rows = predictions(7, 10, {7: FrameDecision.NEW_REPRESENTATIVE, 8: FrameDecision.NEW_REPRESENTATIVE,
                                   9: FrameDecision.NEW_GROUP, 10: FrameDecision.NEW_REPRESENTATIVE})
        records = list(transition_cases(rows, None, {7, 9}, {8}, {7}, "0.95"))
        self.assertEqual([record["frame_index"] for record in records], [7, 8])
        self.assertEqual(records[0]["baseline_decision"], FrameDecision.NEW_GROUP)
        self.assertEqual(records[1]["baseline_decision"], "")
        self.assertTrue(records[1]["manual_new_group"])
