"""Оверлей: линия контроля, светофор, треки, лента инцидентов, FPS."""

from __future__ import annotations

import cv2
import numpy as np

from .events import EventEngine, timecode

C_ALARM = (0, 0, 255)
C_LINE = (0, 255, 255)
C_TEXT = (255, 255, 255)
C_OK = (0, 220, 0)
LIGHT_COLOR = {
    "red": (0, 0, 255),
    "yellow": (0, 220, 255),
    "green": (0, 220, 0),
    "unknown": (160, 160, 160),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALARM_HOLD_SEC = 3.0  # сколько секунд нарушитель остаётся подсвеченным


def _label(frame, text, org, color, scale=0.5, thickness=1):
    """Текст на тёмной подложке — читается на любом фоне."""

    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(frame, (x, y - h - base - 2), (x + w + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (x + 2, y - base), FONT, scale, color, thickness, cv2.LINE_AA
    )


def draw_scene(frame, lines, lights):
    """Линия контроля (красная при красном сигнале) и рамка светофора с состоянием."""

    any_red = any(tl.state == "red" for tl in lights.values())
    for ln in lines.values():
        a, b = tuple(map(int, ln.a)), tuple(map(int, ln.b))
        color = C_ALARM if any_red else C_LINE
        cv2.line(frame, a, b, color, 3, cv2.LINE_AA)
        _label(frame, ln.id.upper(), (b[0] + 8, b[1]), color)
    for tl in lights.values():
        x1, y1, x2, y2 = tl.roi
        color = LIGHT_COLOR[tl.state]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _label(frame, f"LIGHT: {tl.state.upper()}", (x2 + 6, y1 + 22), color, 0.6, 2)


def draw_tracks(frame, tracks, engine: EventEngine, t: float):
    """Рамки, ID и след движения; нарушители — красным."""

    violators = {
        i.track_id: i for i in engine.incidents if t - i.time_sec <= ALARM_HOLD_SEC
    }
    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr.box_xyxy)
        inc = violators.get(tr.track_id)
        color, thick = (C_ALARM, 3) if inc else (C_OK, 2)

        trail = engine.trail(tr.track_id)
        if len(trail) > 1:
            cv2.polylines(
                frame, [np.array(trail, dtype=np.int32)], False, color, 1, cv2.LINE_AA
            )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        ax, ay = map(int, tr.anchor)
        cv2.circle(frame, (ax, ay), 4, color, -1)
        _label(frame, f"#{tr.track_id} {tr.cls} {tr.conf:.2f}", (x1, y1 - 2), color)
        if inc:
            _label(frame, inc.event_type.upper(), (x1, y2 + 18), color)


def draw_hud(frame, fps: float, t: float, engine: EventEngine, feed_size: int = 6):
    """Панель: FPS, число инцидентов и лента последних."""

    rows = [
        (f"FPS {fps:5.1f} | t {timecode(t)[3:]}", C_TEXT),
        (f"incidents: {len(engine.incidents)}", C_TEXT),
    ]
    for inc in engine.incidents[-feed_size:]:
        rows.append((f"{inc.timecode[3:-4]} #{inc.track_id} {inc.event_type}", C_ALARM))

    w, h = 312, 20 * len(rows) + 12
    panel = frame[8 : 8 + h, 8 : 8 + w]
    cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.65, 0, panel)
    for i, (text, color) in enumerate(rows):
        cv2.putText(frame, text, (16, 28 + 20 * i), FONT, 0.5, color, 1, cv2.LINE_AA)
