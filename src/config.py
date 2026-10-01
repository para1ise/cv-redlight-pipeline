from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .traffic_light import TrafficLight
from .zones import LineZone


@dataclass
class ModelConfig:
    weights: str = "yolo11s.pt"
    imgsz: int = 960
    conf: float = 0.25
    device: str | int = 0
    half: bool = False
    classes: list[str] = field(default_factory=lambda: ["car"])
    agnostic_nms: bool = True
    min_hits: int = 3


@dataclass
class RuleConfig:
    """Правило «пересечение линии на красный»."""

    id: str  # тип события в incidents.json
    zone: str  # id линии контроля
    light: str  # id светофора
    direction: str  # to_left | to_right — учитываемое направление пересечения
    classes: list[str]
    grace_sec: float = 0.0  # допуск после включения красного


@dataclass
class Config:
    model: ModelConfig
    tracker: str
    lines: dict[str, LineZone]
    lights: dict[str, TrafficLight]
    rules: list[RuleConfig]


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    lines = {
        z["id"]: LineZone(
            z["id"],
            tuple(map(float, z["points"][0])),
            tuple(map(float, z["points"][1])),
        )
        for z in raw["zones"]
    }
    lights = {
        tl["id"]: TrafficLight(tl["id"], tuple(tl["roi"]), tl.get("min_pixels", 15))
        for tl in raw["traffic_lights"]
    }
    rules = [RuleConfig(**r) for r in raw["rules"]]
    for r in rules:
        if r.zone not in lines:
            raise ValueError(
                f"Правило {r.id} ссылается на несуществующую линию {r.zone}"
            )
        if r.light not in lights:
            raise ValueError(
                f"Правило {r.id} ссылается на несуществующий светофор {r.light}"
            )

    return Config(
        model=ModelConfig(**raw.get("model", {})),
        tracker=raw.get("tracker", "bytetrack.yaml"),
        lines=lines,
        lights=lights,
        rules=rules,
    )
