from __future__ import annotations

import re

OPEN = "<tag>"
_FULL = re.compile(r"<tag>\s*(\d{1,6})\s*</tag>")
_PARTIAL = re.compile(r"<tag>\s*\d{0,6}\s*(?:<(?:/(?:t(?:a(?:g)?)?)?)?)?")


class TagFilter:
    def __init__(self) -> None:
        self._held = ""
        self.tag: int | None = None

    def feed(self, text: str) -> str:
        self._held += text
        out: list[str] = []
        while self._held:
            start = self._held.find(OPEN)
            if start < 0:
                keep = _open_prefix_len(self._held)
                out.append(self._held[: len(self._held) - keep])
                self._held = self._held[len(self._held) - keep:]
                break
            out.append(self._held[:start])
            self._held = self._held[start:]
            match = _FULL.match(self._held)
            if match:
                self.tag = int(match.group(1))
                self._held = self._held[match.end():]
                continue
            if _PARTIAL.fullmatch(self._held):
                break
            out.append(OPEN)
            self._held = self._held[len(OPEN):]
        return "".join(out)

    def finish(self) -> str:
        rest, self._held = self._held, ""
        return rest


def _open_prefix_len(text: str) -> int:
    for size in range(min(len(OPEN) - 1, len(text)), 0, -1):
        if OPEN.startswith(text[-size:]):
            return size
    return 0
