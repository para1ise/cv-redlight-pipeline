"""Запись фрагмента трансляции в файл.

    python utils/record_stream.py --url https://www.youtube.com/watch?v=1CI685iPIUQ --seconds 600 --out data/day.ts

MPEG-TS записывает после обрыва связи.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.source import resolve_url

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--out", default="data/record.ts")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stream = resolve_url(args.url)
    print(f"Пишу {args.seconds} с в {args.out} ...")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            stream,
            "-t",
            str(args.seconds),
            "-c",
            "copy",
            "-an",
            "-f",
            "mpegts",
            args.out,
        ],
        check=True,
    )
    print("Готово:", args.out)
