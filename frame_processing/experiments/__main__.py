"""CLI for capture and offline SSIM threshold experiments."""

import argparse
from pathlib import Path
import sys

from interface_spec.contracts import CaptureRegion

from ..capture import MSSCapture
from ..synthetic import SyntheticCapture
from .capture import DEFAULT_ROOT, capture_dataset, load_dataset
from .evaluate import compare_frame, evaluate_run
from .labels import ensure_labels, review_frame
from .preservation import LabelRequirements, ensure_events
from .progress import SweepProgress
from .replay import load_sweep, parse_threshold_pairs, parse_thresholds, run_sweep
from .window_evaluation import evaluate_windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="collect lossless PNGs at 1 FPS")
    capture.add_argument("--name", required=True)
    capture.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
                         required=True)
    capture.add_argument("--seconds", type=float, required=True)
    capture.add_argument("--synthetic", action="store_true", help="test without acquiring the desktop")
    sweep = commands.add_parser("sweep", help="replay the same capture for each threshold")
    sweep.add_argument("dataset", type=Path)
    thresholds = sweep.add_mutually_exclusive_group()
    thresholds.add_argument("--thresholds", help="comma-separated threshold list")
    thresholds.add_argument("--range", nargs=3, metavar=("START", "STOP", "STEP"))
    sweep.add_argument("--group-thresholds", help="comma-separated P/C thresholds")
    sweep.add_argument("--representative-thresholds", help="comma-separated R/C thresholds")
    sweep.add_argument("--comparison-max-side", type=int, default=960,
                       help="longest side in comparison; use 0 for no resize")
    sweep.add_argument("--write-representatives", action="store_true", help="also save representative JPEGs")
    sweep.add_argument("--start-frame", type=int, help="first original frame index, inclusive")
    sweep.add_argument("--end-frame", type=int, help="last original frame index, inclusive")
    sweep.add_argument("--no-progress", action="store_true", help="hide the sweep progress on stderr")
    review = commands.add_parser("review", help="prepare labels and optionally view source frames side by side")
    review.add_argument("dataset", type=Path)
    review.add_argument("--frame", type=int)
    review.add_argument("--reference", type=int)
    review.add_argument("--run", type=Path, help="limit viewing to the frames selected by this sweep")
    evaluate = commands.add_parser("evaluate", help="score the existing sweep with current labels")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("run", type=Path)
    evaluate.add_argument("--checks", type=Path, help="edited evidence_to_review.csv from an evaluation")
    defaults = LabelRequirements()
    evaluate.add_argument("--min-labeled-frames", type=int, default=defaults.min_labeled_frames,
                          help="minimum labeled frames for candidate selection (default: %(default)s)")
    evaluate.add_argument("--min-labeling-ratio", type=float, default=defaults.min_labeling_ratio,
                          help="minimum labeled fraction, 0 to 1 (default: %(default)s)")
    evaluate.add_argument("--min-labels-per-binary-class", type=int, default=defaults.min_labels_per_binary_class,
                          help="minimum labels in each of meaningful change and reused (default: %(default)s)")
    windows = commands.add_parser("evaluate-windows", help="evaluate annotated windows against an existing sweep")
    windows.add_argument("dataset", type=Path)
    windows.add_argument("run", type=Path)
    windows.add_argument("--start-frame", type=int, help="first original frame to evaluate, inclusive")
    windows.add_argument("--end-frame", type=int, help="last original frame to evaluate, inclusive")
    windows.add_argument("--required-frames", action="store_true",
                         help="use required_frames.csv with one-to-one frame-or-predecessor matching")
    windows.add_argument("--baseline", type=Path, help="existing single-threshold sweep for comparison")
    windows.add_argument("--transition-threshold", default="0.95",
                         help="single baseline threshold to flag possible group transitions (default: 0.95)")
    compare = commands.add_parser("compare", help="compare all thresholds at a single frame index")
    compare.add_argument("dataset", type=Path)
    compare.add_argument("run", type=Path)
    compare.add_argument("--frame", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "capture":
            path = capture_dataset(DEFAULT_ROOT, args.name, CaptureRegion(*args.region), args.seconds,
                                   SyntheticCapture() if args.synthetic else MSSCapture())
            dataset = load_dataset(path)
            ensure_labels(dataset)
            ensure_events(dataset)
            print(path)
        elif args.command == "sweep":
            group_values, representative_values = args.group_thresholds, args.representative_thresholds
            if group_values is not None or representative_values is not None:
                if group_values is None or representative_values is None or args.thresholds or args.range:
                    parser.error("provide both dual threshold lists, without --thresholds or --range")
                candidates = parse_threshold_pairs(group_values, representative_values)
            else:
                candidates = parse_thresholds(args.thresholds, args.range)
            progress = None if args.no_progress else SweepProgress(sys.stderr)
            try:
                path = run_sweep(load_dataset(args.dataset), candidates,
                                 max_side=args.comparison_max_side or None,
                                 write_representatives=args.write_representatives,
                                 start_frame=args.start_frame, end_frame=args.end_frame, progress=progress)
            finally:
                if progress is not None:
                    progress.close()
            print(path)
        elif args.command == "review":
            if args.reference is not None and args.frame is None:
                parser.error("--reference requires --frame")
            dataset = load_dataset(args.dataset)
            selection = load_sweep(dataset, args.run) if args.run is not None else None
            print(ensure_labels(dataset))
            print(ensure_events(dataset))
            if args.frame is not None:
                if selection is None:
                    print(review_frame(dataset, args.frame, args.reference))
                else:
                    print(review_frame(dataset, args.frame, args.reference,
                                       start_frame=selection.start, end_frame=selection.end))
        elif args.command == "evaluate":
            requirements = LabelRequirements(args.min_labeled_frames, args.min_labeling_ratio,
                                             args.min_labels_per_binary_class)
            print(evaluate_run(load_dataset(args.dataset), args.run, args.checks,
                               label_requirements=requirements))
        elif args.command == "evaluate-windows":
            print(evaluate_windows(load_dataset(args.dataset), args.run,
                                   start_frame=args.start_frame, end_frame=args.end_frame,
                                   required_frames=args.required_frames, baseline=args.baseline,
                                   transition_threshold=args.transition_threshold))
        elif args.command == "compare":
            print(compare_frame(load_dataset(args.dataset), args.run, args.frame))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"{args.command}: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, f"{args.command}: interrupted\n")


if __name__ == "__main__":
    main()
