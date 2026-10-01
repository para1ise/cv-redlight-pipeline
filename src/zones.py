"""Геометрия контрольной линии (tripwire).

Модуль не зависит от детектора и трекера — только чистая геометрия.
"""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]


def cross(o: Point, a: Point, b: Point) -> float:
    """Векторное произведение (a - o) x (b - o).

    Знак показывает, с какой стороны от направленного отрезка o->a лежит точка b.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Пересекаются ли отрезки p1-p2 и q1-q2 (строгое пересечение).

    Отрезки пересекаются, если концы каждого лежат по разные стороны от другого.
    """
    d1 = cross(q1, q2, p1)
    d2 = cross(q1, q2, p2)
    d3 = cross(p1, p2, q1)
    d4 = cross(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


@dataclass
class LineZone:
    """Tripwire: направленный отрезок a -> b."""

    id: str
    a: Point
    b: Point

    def crossing(self, prev: Point, curr: Point) -> str | None:
        """Направление пересечения линии при перемещении prev -> curr.

        Возвращает "to_left" / "to_right" — на какую сторону от a->b
        перешёл объект, или None, если линия не пересечена.
        """
        if not segments_intersect(prev, curr, self.a, self.b):
            return None
        return "to_left" if cross(self.a, self.b, curr) > 0 else "to_right"
