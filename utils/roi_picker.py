"""Разметка контрольной линии мышью на кадре видео. Результат — блок `zones:` для конфига.

    python utils/roi_picker.py --source data/demo_input.mp4 --time 12

Управление:
    ЛКМ  — поставить точку (первая — a, вторая — b)
    ПКМ  — убрать последнюю точку
    s    — вывести блок для конфига и выйти
    Esc  — выйти без результата
"""

import argparse

import cv2

WINDOW = "roi_picker"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument(
        "--time", type=float, default=0.0, help="секунда видео, с которой взять кадр"
    )
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    ok, base = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Не удалось прочитать кадр из {args.source}")

    points: list[tuple[int, int]] = []

    def redraw():
        img = base.copy()
        for p in points:
            cv2.circle(img, p, 5, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            img,
            "LMB: point  RMB: undo  s: print & quit  Esc: quit",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow(WINDOW, img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
        redraw()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        if key == ord("s") and len(points) == 2:
            print("zones:\n  - id: stop_line\n    points:", [list(p) for p in points])
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
