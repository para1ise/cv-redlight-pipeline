"""Событийный движок: проезд контрольной линии на красный сигнал.

На вход — список Track за кадр, время кадра и состояния светофоров,
на выход — инциденты. Про нейросеть и видео модуль ничего не знает.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .config import Config
from .tracking import Track

LOST_TRACK_SEC = 2.0  # трек не виден дольше — забываем его историю
LINE_COOLDOWN_SEC = 1.0  # защита от повторного срабатывания на дрожании bbox


def timecode(t: float) -> str:
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


@dataclass
class Incident:
    timecode: str
    event_type: str
    track_id: int
    time_sec: float  # для оверлея, в JSON не выгружается

    def to_dict(self) -> dict:
        return {
            "timecode": self.timecode,
            "event_type": self.event_type,
            "track_id": self.track_id,
        }


class EventEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.incidents: list[Incident] = []
        self._history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=30)
        )  # (t, точка опоры)
        self._last_fired: dict[tuple[str, int], float] = {}

    def update(
        self, t: float, tracks: list[Track], signals: dict[str, tuple[str, float]]
    ) -> list[Incident]:
        """Обрабатывает кадр, возвращает инциденты этого кадра.

        signals — состояния светофоров: {id: (цвет, с какого момента)}.
        """

        new: list[Incident] = []
        for tr in tracks:
            hist = self._history[tr.track_id]
            prev = hist[-1][1] if hist else None
            hist.append((t, tr.anchor))
            if prev is None:
                continue
            for rule in self.cfg.rules:
                if tr.cls in rule.classes and self._red_light_crossing(
                    rule, tr, prev, t, signals
                ):
                    inc = Incident(timecode(t), rule.id, tr.track_id, t)
                    self.incidents.append(inc)
                    new.append(inc)
        self._drop_lost(t)
        return new

    def trail(self, track_id: int) -> list[tuple[float, float]]:
        """Последние положения точки опоры (для отрисовки следа)."""

        return [a for _, a in self._history.get(track_id, ())]

    def _red_light_crossing(self, rule, tr: Track, prev, t: float, signals) -> bool:
        # 1. линия пересечена между соседними кадрами, причём в нужном направлении
        direction = self.cfg.lines[rule.zone].crossing(prev, tr.anchor)
        if direction is None or direction != rule.direction:
            return False
        # 2. сигнал красный и допуск после его включения истёк
        state, since = signals.get(rule.light, ("unknown", 0.0))
        if state != "red" or t - since < rule.grace_sec:
            return False
        # 3. один трек — одно срабатывание в пределах LINE_COOLDOWN_SEC
        key = (rule.id, tr.track_id)
        if t - self._last_fired.get(key, -1e9) < LINE_COOLDOWN_SEC:
            return False
        self._last_fired[key] = t
        return True

    def _drop_lost(self, t: float):
        for tid in [
            tid for tid, h in self._history.items() if t - h[-1][0] > LOST_TRACK_SEC
        ]:
            del self._history[tid]
