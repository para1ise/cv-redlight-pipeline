from __future__ import annotations

import cv2


def is_stream(source: str) -> bool:
    return source.startswith(("rtsp://", "rtmp://", "http://", "https://"))


def resolve_url(source: str, max_height: int = 720) -> str:
    """Ссылку на страницу YouTube превращаем в прямой URL потока (HLS) через yt-dlp.

    RTSP/HLS-адреса и пути к файлам возвращаются как есть.
    """

    if "youtube.com" not in source and "youtu.be" not in source:
        return source
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": f"bv*[height<={max_height}]/b[height<={max_height}]",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(source, download=False)["url"]


def open_source(source: str) -> tuple[cv2.VideoCapture, float, bool]:
    """Возвращает (capture, fps, live)."""

    live = is_stream(source)
    cap = cv2.VideoCapture(resolve_url(source))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 120:  # 0 / NaN / мусор у некоторых потоков
        fps = 30.0
    return cap, fps, live
