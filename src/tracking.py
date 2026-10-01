"""Детекция (YOLO) + трекинг (ByteTrack)."""

from __future__ import annotations

from dataclasses import dataclass

from ultralytics import YOLO

from .config import ModelConfig


@dataclass
class Track:
    track_id: int
    cls: str
    conf: float
    box_xyxy: tuple[float, float, float, float]

    @property
    def anchor(self) -> tuple[float, float]:
        """Точка опоры — середина нижней грани bbox - колеса"""

        x1, _, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2, y2)


class ObjectTracker:
    def __init__(self, cfg: ModelConfig, tracker_cfg: str):
        self.cfg = cfg
        self.tracker_cfg = tracker_cfg
        self.model = YOLO(cfg.weights)
        self.names: dict[int, str] = self.model.names
        # фильтруем классы на уровне модели, чтобы лишние не попадали в трекер
        self.class_ids = [i for i, n in self.names.items() if n in cfg.classes]
        self._hits: dict[int, int] = {}  # сколько кадров трек уже наблюдается

    def update(self, frame) -> list[Track]:
        # persist=True — для сохранения состояния ByteTrack между вызовами
        result = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            classes=self.class_ids,
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            half=self.cfg.half,
            agnostic_nms=self.cfg.agnostic_nms,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        tracks = []
        for tid, c, conf, xyxy in zip(
            boxes.id.tolist(),
            boxes.cls.tolist(),
            boxes.conf.tolist(),
            boxes.xyxy.tolist(),
        ):
            tid = int(tid)
            self._hits[tid] = self._hits.get(tid, 0) + 1
            # короткоживущие треки (блики фар, ложные детекции) наружу не отдаём
            if self._hits[tid] >= self.cfg.min_hits:
                tracks.append(
                    Track(tid, self.names[int(c)], float(conf), tuple(map(float, xyxy)))
                )
        return tracks
