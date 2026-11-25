"""Open a local labeling window for the representative PNGs from a saved SSIM sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2

from interface_spec.contracts import FrameType

from .classification_data import LectureSource, prepare, read_image
from .labeling_session import LabelingSession, editing_lock


LABEL_KEYS = {"1": FrameType.TEXT, "2": FrameType.VISUAL, "3": FrameType.UNKNOWN}
LECTURE_NAMES = {"static": "정적 슬라이드", "handwriting": "필기", "animation": "애니메이션"}


class LabelingWindow:
    def __init__(self, root: tk.Tk, session: LabelingSession) -> None:
        self.root, self.session = root, session
        self._pixels = None
        self._photo = None
        self._resize_job: str | None = None
        self._held_labels: set[str] = set()
        self._scopes = (None, *session.lectures)
        root.title("대표 프레임 라벨링")
        width = min(1450, int(root.winfo_screenwidth() * .93))
        height = min(920, int(root.winfo_screenheight() * .87))
        root.geometry(f"{width}x{height}")
        root.minsize(640, 440)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#f4f6fa")
        style.configure("App.TLabel", background="#f4f6fa", foreground="#36465e", font=("Malgun Gothic", 10))
        style.configure("Title.TLabel", background="#f4f6fa", foreground="#152b46",
                        font=("Malgun Gothic", 16, "bold"))
        style.configure("Error.TLabel", background="#f4f6fa", foreground="#a52a22", font=("Malgun Gothic", 10))
        style.configure("TButton", font=("Malgun Gothic", 10), padding=(12, 8))
        style.configure("Selected.TButton", background="#cde4fc", foreground="#113e6a",
                        font=("Malgun Gothic", 10, "bold"), padding=(12, 8))
        style.configure("Review.Horizontal.TProgressbar", background="#2373b8", troughcolor="#e0e8f3")

        app = ttk.Frame(root, style="App.TFrame", padding=16)
        app.pack(fill="both", expand=True)
        app.columnconfigure(0, weight=1)
        app.rowconfigure(2, weight=1)

        header = ttk.Frame(app, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="대표 프레임 라벨링", style="Title.TLabel").pack(side="left")
        self.scope_box = ttk.Combobox(
            header, state="readonly", width=21,
            values=("전체 강의", *(LECTURE_NAMES.get(lecture, lecture) for lecture in session.lectures)),
        )
        self.scope_box.pack(side="right")
        self.scope_box.current(self._scopes.index(session.scope))
        self.scope_box.bind("<<ComboboxSelected>>", self._select_scope)

        progress = ttk.Frame(app, style="App.TFrame", padding=(0, 8, 0, 10))
        progress.grid(row=1, column=0, sticky="ew")
        self.frame_info = ttk.Label(progress, style="App.TLabel")
        self.frame_info.pack(side="left")
        self.progress_text = ttk.Label(progress, style="App.TLabel")
        self.progress_text.pack(side="right")
        self.progress_bar = ttk.Progressbar(progress, style="Review.Horizontal.TProgressbar",
                                           length=110, maximum=100, mode="determinate")
        self.progress_bar.pack(side="right", padx=(10, 8))

        self.canvas = tk.Canvas(app, background="#101923", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._schedule_paint)

        footer = ttk.Frame(app, style="App.TFrame", padding=(0, 12, 0, 0))
        footer.grid(row=3, column=0, sticky="ew")
        self.previous = ttk.Button(footer, text="← 이전", command=lambda: self._move(-1))
        self.previous.pack(side="left")
        self.next = ttk.Button(footer, text="다음 →", command=lambda: self._move(1))
        self.next.pack(side="left", padx=(6, 14))
        self.label_buttons = {}
        for key, label in LABEL_KEYS.items():
            button = ttk.Button(footer, text=f"{key}  {label.value}", command=lambda kind=label: self._assign(kind))
            button.pack(side="left", padx=(0, 6))
            self.label_buttons[label] = button
        self.label_info = ttk.Label(footer, style="App.TLabel")
        self.label_info.pack(side="right")
        self.status = ttk.Label(app, style="Error.TLabel", wraplength=590)
        self.status.grid(row=4, column=0, sticky="w")

        root.bind("<KeyPress>", self._on_key)
        root.bind("<KeyRelease>", self._on_release)
        root.bind("<FocusOut>", self._on_blur)
        self._show_frame()

    def _on_key(self, event):
        key = event.keysym.removeprefix("KP_")
        if key in LABEL_KEYS:
            if key not in self._held_labels:
                self._held_labels.add(key)
                self._assign(LABEL_KEYS[key])
            return "break"
        if key in ("Left", "Right"):
            self._move(-1 if key == "Left" else 1)
            return "break"

    def _on_release(self, event) -> None:
        self._held_labels.discard(event.keysym.removeprefix("KP_"))

    def _on_blur(self, _event) -> None:
        self._held_labels.clear()

    def _select_scope(self, _event=None) -> None:
        self.session.select(self._scopes[self.scope_box.current()])
        self.canvas.focus_set()
        self._after_navigation()

    def _move(self, offset: int) -> None:
        if self.session.move(offset):
            self._after_navigation()

    def _assign(self, label: FrameType) -> None:
        if self._pixels is None:
            return
        try:
            self.session.assign(label)
        except OSError as exc:
            self.status.configure(text=f"라벨을 저장하지 못했습니다. {exc}")
            return
        self._after_navigation()

    def _after_navigation(self) -> None:
        self._show_frame()
        try:
            self.session.save_cursor()
        except OSError as exc:
            previous = self.status.cget("text")
            self.status.configure(text=f"{previous}\n작업 위치 저장 실패. {exc}".lstrip())

    def _show_frame(self) -> None:
        sample = self.session.current
        done, count = self.session.progress
        total_done, total = self.session.total_progress
        progress = 100 * done / count
        self.progress_bar["value"] = progress
        summary = f"완료 {done:,}/{count:,} · {progress:.1f}%"
        if self.session.scope is not None:
            summary += f"  |  전체 {total_done:,}/{total:,}"
        self.progress_text.configure(text=summary)
        self.frame_info.configure(
            text=f"{sample.lecture}  ·  원본 {sample.frame_index:,}번  ·  {self.session.position + 1:,}/{count:,}",
        )
        label = self.session.current_label
        self.label_info.configure(text=f"현재  {label or '미판정'}")
        self.previous.state(["!disabled"] if self.session.position else ["disabled"])
        self.next.state(["!disabled"] if self.session.position + 1 < count else ["disabled"])
        self.status.configure(text="")
        try:
            self._pixels = read_image(self.session.output, sample)
        except (OSError, ValueError, cv2.error) as exc:
            self._pixels = None
            self.status.configure(text=f"원본 PNG를 열 수 없습니다. {exc}")
        for kind, button in self.label_buttons.items():
            button.configure(style="Selected.TButton" if kind.value == label else "TButton")
            button.state(["!disabled"] if self._pixels is not None else ["disabled"])
        self._paint_image()

    def _schedule_paint(self, _event) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(80, self._paint_image)

    def _paint_image(self) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
            self._resize_job = None
        self.canvas.delete("all")
        self._photo = None
        if self._pixels is None:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        h, w = self._pixels.shape[:2]
        scale = min(width / w, height / h)
        display = cv2.resize(self._pixels, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        ok, png = cv2.imencode(".png", display, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if ok:
            self._photo = tk.PhotoImage(master=self.root, data=png.tobytes(), format="png")
            self.canvas.create_image(width // 2, height // 2, image=self._photo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new or existing labeling project")
    parser.add_argument("--source", action="append", nargs=4,
                        metavar=("LECTURE_ID", "DATASET", "SWEEP", "CANDIDATE"),
                        help="repeat for each lecture when creating a new project")
    parser.add_argument("--prepare-only", action="store_true", help="write a new manifest and label sheet")
    args = parser.parse_args()
    if args.prepare_only and not args.source:
        parser.error("--prepare-only needs at least one --source")
    try:
        if args.source:
            if args.output.exists():
                parser.error("output already exists; omit --source to resume, or choose a new output")
            sources = [LectureSource(lecture, Path(dataset), Path(sweep), candidate)
                       for lecture, dataset, sweep, candidate in args.source]
            samples = prepare(args.output, sources, count=None)
            print(f"{len(samples)} representative frames: {args.output / 'labels.csv'}")
        if not args.prepare_only:
            with editing_lock(args.output):
                session = LabelingSession(args.output.resolve())
                root = tk.Tk()
                LabelingWindow(root, session)
                root.mainloop()
    except (OSError, ValueError, tk.TclError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
