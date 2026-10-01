"""Экспорт результатов: incidents.json и перекодирование видео в H.264."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .events import Incident


def write_incidents(path: str | Path, incidents: list[Incident]):
    """Перечень инцидентов: таймкод, тип события, ID объекта."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([i.to_dict() for i in incidents], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def to_h264(src: str | Path, dst: str | Path) -> bool:
    """OpenCV пишет mp4v, который не играет в браузере/GitHub — перекодируем в H.264."""

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                str(dst),
            ],
            check=True,
        )
        return True
    except (
        Exception
    ) as e:  # на случай если ffmpeg не установлен — оставляем исходный файл
        print(f"[warn] перекодирование в H.264 не выполнено: {e}")
        return False
