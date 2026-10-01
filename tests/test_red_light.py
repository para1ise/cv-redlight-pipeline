import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, ModelConfig, RuleConfig
from src.events import EventEngine
from src.tracking import Track
from src.traffic_light import TrafficLight, classify_light
from src.zones import LineZone


def lamp(bgr) -> np.ndarray:
    """Тёмный кадр 40x100 с яркой лампой заданного цвета."""
    img = np.zeros((40, 100, 3), dtype=np.uint8)
    img[10:20, 40:50] = bgr
    return img


def test_classify_colors():
    assert classify_light(lamp((0, 0, 255)))[0] == "red"
    assert classify_light(lamp((0, 255, 0)))[0] == "green"
    assert classify_light(lamp((0, 220, 255)))[0] == "yellow"
    assert classify_light(np.zeros((40, 100, 3), dtype=np.uint8))[0] == "unknown"


def test_light_debounce_and_since():
    tl = TrafficLight("l", (0, 0, 100, 40), debounce_frames=3)
    red, green = lamp((0, 0, 255)), lamp((0, 255, 0))
    for i in range(5):
        tl.update(green, i * 0.1)
    assert tl.state == "green"
    # один "красный" кадр состояние не меняет
    tl.update(red, 0.5)
    assert tl.state == "green"
    tl.update(green, 0.6)
    for i in range(3):
        tl.update(red, 1.0 + i * 0.1)
    assert tl.state == "red"
    assert tl.since == 1.0  # начало серии, а не момент подтверждения


def make_engine(grace=0.3):
    rule = RuleConfig(
        id="red_light_violation",
        zone="stop",
        light="main",
        direction="to_right",
        classes=["car"],
        grace_sec=grace,
    )
    cfg = Config(
        model=ModelConfig(),
        tracker="",
        lines={"stop": LineZone("stop", (0, 100), (200, 100))},
        lights={},
        rules=[rule],
    )
    return EventEngine(cfg)


def car(y):
    # точка опоры — середина нижней грани: (100, y)
    return Track(1, "car", 0.9, (80, y - 40, 120, y))


def drive(engine, ys, signal):
    out = []
    for i, y in enumerate(ys):
        out += engine.update(10.0 + i * 0.1, [car(y)], {"main": signal})
    return out


UP = (130, 115, 95, 80)  # от камеры, пересекает линию y=100
DOWN = (80, 95, 115, 130)  # к камере


def test_violation_on_red():
    incidents = drive(make_engine(), UP, ("red", 5.0))
    assert len(incidents) == 1
    assert incidents[0].to_dict() == {
        "timecode": "00:00:10.200",
        "event_type": "red_light_violation",
        "track_id": 1,
    }


def test_no_violation_on_green():
    assert drive(make_engine(), UP, ("green", 5.0)) == []


def test_grace_period():
    # красный включился за 0.1 с до пересечения — в пределах допуска 0.3 с
    assert drive(make_engine(grace=0.3), UP, ("red", 10.1)) == []


def test_opposite_direction_ignored():
    assert drive(make_engine(), DOWN, ("red", 5.0)) == []
