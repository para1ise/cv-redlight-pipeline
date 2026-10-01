"""CV-пайплайн: детекция -> трекинг -> проезд на красный -> оверлей + incidents.json.

python main.py --source input.mp4
python main.py --source https://www.youtube.com/watch?v=1CI685iPIUQ --max-seconds 45
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.config import load_config
from src.events import EventEngine
from src.export import to_h264, write_incidents
from src.render import draw_hud, draw_scene, draw_tracks
from src.source import open_source
from src.tracking import ObjectTracker


def parse_args():
    ap = argparse.ArgumentParser(description="Red light violation detection pipeline")
    ap.add_argument(
        "--source",
        required=True,
        help="видеофайл, RTSP/HLS URL или ссылка на YouTube-трансляцию",
    )
    ap.add_argument("--config", default="configs/yokohama.yaml")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="обработать не более N секунд (0 — до конца)",
    )
    ap.add_argument("--show", action="store_true", help="показывать окно с результатом")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap, fps, live = open_source(args.source)
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    tracker = ObjectTracker(cfg.model, cfg.tracker)
    engine = EventEngine(cfg)

    raw_path = out_dir / "annotated_raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    frame_idx, proc_time, fps_smooth = 0, 0.0, 0.0
    wall_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # файл: время по номеру кадра (воспроизводимо); live: реальное время
        t = (time.perf_counter() - wall_start) if live else frame_idx / fps
        if args.max_seconds and t >= args.max_seconds:
            break

        t0 = time.perf_counter()
        tracks = tracker.update(frame)  # 1. детекция + трекинг
        signals = {
            tl.id: (tl.update(frame, t), tl.since) for tl in cfg.lights.values()
        }  # 2. светофор
        for inc in engine.update(t, tracks, signals):  # 3. события
            print(f"[{inc.timecode}] {inc.event_type} track #{inc.track_id}")

        draw_scene(frame, cfg.lines, cfg.lights)  # 4. оверлей
        draw_tracks(frame, tracks, engine, t)
        dt = time.perf_counter() - t0
        proc_time += dt
        fps_smooth = 1 / dt if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 / dt
        draw_hud(frame, fps_smooth, t, engine)

        writer.write(frame)
        if args.show:
            cv2.imshow("cv-roi-incidents", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc
                break
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    write_incidents(out_dir / "incidents.json", engine.incidents)
    if to_h264(raw_path, out_dir / "annotated.mp4"):
        raw_path.unlink()

    avg_fps = frame_idx / proc_time if proc_time else 0.0
    print(f"\nКадров: {frame_idx}, средний FPS обработки: {avg_fps:.1f}")
    print(f"Инцидентов: {len(engine.incidents)} -> {out_dir / 'incidents.json'}")


if __name__ == "__main__":
    main()
