"""Состояние светофора по цвету в заданной прямоугольной зоне кадра.

Считаем яркие насыщенные пиксели каждого цвета в HSV и берём преобладающий.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

HUE_RANGES = {
    "red": [(0, 12), (168, 179)],
    "yellow": [(15, 35)],
    "green": [(60, 100)],
}


def classify_light(
    roi_bgr: np.ndarray,
    min_pixels: int = 15,
    min_value: int = 200,
    min_saturation: int = 120,
) -> tuple[str, dict[str, int]]:
    """Возвращает (цвет | "unknown", число пикселей по цветам)."""

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (v >= min_value) & (s >= min_saturation)
    counts = {
        color: int(sum((lit & (h >= lo) & (h <= hi)).sum() for lo, hi in ranges))
        for color, ranges in HUE_RANGES.items()
    }
    best = max(counts, key=counts.get)
    return (best if counts[best] >= min_pixels else "unknown"), counts


@dataclass
class TrafficLight:
    id: str
    roi: tuple[int, int, int, int]  # x1, y1, x2, y2
    min_pixels: int = 15
    debounce_frames: int = 3  # смена состояния подтверждается N кадрами подряд

    state: str = "unknown"
    since: float = 0.0  # время, когда установилось текущее состояние
    _candidate: str = "unknown"
    _streak: int = 0
    _candidate_t: float = 0.0

    def update(self, frame: np.ndarray, t: float) -> str:
        x1, y1, x2, y2 = self.roi
        observed, _ = classify_light(frame[y1:y2, x1:x2], self.min_pixels)
        # "unknown" (лампа на мгновение не видна) состояние не меняет
        if observed == "unknown":
            return self.state
        if observed == self.state:
            # серия кандидата прервалась — начинаем отсчёт заново
            self._candidate, self._streak = self.state, 0
            return self.state
        if observed != self._candidate:
            self._candidate, self._streak, self._candidate_t = observed, 0, t
        self._streak += 1
        if self._streak >= self.debounce_frames:
            self.state, self.since = observed, self._candidate_t
            self._streak = 0
        return self.state
