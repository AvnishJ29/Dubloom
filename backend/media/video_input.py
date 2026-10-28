"""Fetch a source video from any URL or an uploaded file and normalise it for dubbing."""

from __future__ import annotations

import ipaddress
import logging
import os
import json
import mimetypes
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse, unquote

from backend.config import settings

import httpx

MEDIA_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
    ".mpg",
    ".mpeg",
    ".ts",
    ".flv",
    ".wmv",
    ".3gp",
    ".ogv",
    ".mts",
    ".m2ts",
    ".mxf",
}
HLS_TYPES = {"application/vnd.apple.mpegurl", "application/x-mpegurl", "audio/mpegurl"}


class SourceError(RuntimeError):
    """The URL or file cannot be used as a dubbing source."""


logger = logging.getLogger(__name__)


@dataclass
class SourceInfo:
    """A prepared source path with its title, origin metadata, and section state."""

    path: Path
    title: str
    origin: str
    meta: dict = field(
        default_factory=dict
    )  # categories, tags, channel, description (yt-dlp sources)
    sectioned: bool = False  # the file already holds only the requested part (range download)


def run(*args, cwd=None, env=None):
    """Run a media command and return stdout, raising with stderr on failure."""
    result = subprocess.run(
        [str(x) for x in args], cwd=cwd, env=env, capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr[-1600:]}")
    return result.stdout


def probe(path: Path) -> dict:
    """Return normalized video/audio metadata from ffprobe."""
    raw = run("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path)
    data = json.loads(raw)
    video = next(
        (
            s
            for s in data.get("streams", [])
            if s.get("codec_type") == "video"
            and s.get("disposition", {}).get("attached_pic", 0) == 0
        ),
        None,
    )
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})
    try:
        seconds = float(fmt.get("duration") or (video or {}).get("duration") or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    return {
        "duration": seconds,
        "format_name": fmt.get("format_name", ""),
        "has_video": video is not None,
        "has_audio": audio is not None,
        "video_codec": (video or {}).get("codec_name"),
        "audio_codec": (audio or {}).get("codec_name"),
        "pix_fmt": (video or {}).get("pix_fmt"),
        "width": int((video or {}).get("width") or 0),
        "height": int((video or {}).get("height") or 0),
        "size": int(fmt.get("size") or 0),
    }


def duration(path: Path) -> float:
    """Container duration of a media file in seconds, from ffprobe."""
    raw = run("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path)
    return float(json.loads(raw)["format"]["duration"])


def validate_url(url: str) -> str:
    """Validate an HTTP(S) source URL and reject detected local-network addresses."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SourceError("Enter a valid http or https video URL")
    host = parsed.hostname.lower()
    if host in {"localhost", "0.0.0.0"} or host.endswith(".local") or host.endswith(".internal"):
        raise SourceError("Local network addresses are not allowed")
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
            address = ipaddress.ip_address(sockaddr[0])
            if (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_reserved
            ):
                raise SourceError("Local network addresses are not allowed")
    except socket.gaierror as exc:
        raise SourceError(f"Cannot resolve host {host}") from exc
    return parsed.geturl()


def _guess_extension(url: str, content_type: str) -> str:
    path_ext = Path(unquote(urlparse(url).path)).suffix.lower()
    if path_ext in MEDIA_EXTENSIONS:
        return path_ext
    guessed = (
        mimetypes.guess_extension(content_type.split(";")[0].strip()) if content_type else None
    )
    if guessed and guessed.lower() in MEDIA_EXTENSIONS:
        return guessed.lower()
    return ".mp4"


def _inspect_url(url: str) -> tuple[str, int, str]:
    """Return (content type, content length, final URL) without downloading the body."""
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Dubloom/1.0", "Range": "bytes=0-0"}
    with httpx.Client(follow_redirects=True, timeout=20) as client:
        try:
            response = client.get(url, headers=headers)
        except httpx.HTTPError:
            return "", 0, url
        content_type = response.headers.get("content-type", "").lower()
        length = 0
        content_range = response.headers.get("content-range", "")
        match = re.search(r"/(\d+)$", content_range)
        if match:
            length = int(match.group(1))
        elif response.status_code == 200 and response.headers.get("content-length"):
            length = int(response.headers["content-length"])
        response.close()
        return content_type, length, str(response.url)


def _is_direct_media(url: str, content_type: str) -> bool:
    """True when the URL serves a media file itself rather than a page or a playlist."""
    if content_type.startswith(("video/", "audio/")):
        return True
    ext = Path(unquote(urlparse(url).path)).suffix.lower()
    if ext in MEDIA_EXTENSIONS and (
        content_type.startswith("application/octet-stream")
        or content_type.startswith("binary/")
        or not content_type
    ):
        return True
    return False


def _download_direct(url: str, destination: Path, max_bytes: int, progress=None) -> Path:
    """Stream a media URL to `destination`, enforcing `max_bytes`.

    Reports the downloaded share to `progress` when the size is known. Raises SourceError on an
    HTTP error, an oversized file or an empty body.
    """
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Dubloom/1.0"}
    received = 0
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30, read=120)) as client:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code >= 400:
                raise SourceError(f"The media URL returned HTTP {response.status_code}")
            total = int(response.headers.get("content-length") or 0)
            if total and total > max_bytes:
                raise SourceError(
                    f"The media file is larger than the {max_bytes // (1024 * 1024)} MB limit"
                )
            with destination.open("wb") as output:
                for chunk in response.iter_bytes(1024 * 512):
                    output.write(chunk)
                    received += len(chunk)
                    if received > max_bytes:
                        raise SourceError(
                            f"The media file is larger than the {max_bytes // (1024 * 1024)} MB limit"
                        )
                    if progress and total:
                        progress(min(0.99, received / total))
    if received == 0:
        raise SourceError("The media URL returned an empty file")
    return destination


def _download_stream(url: str, destination: Path, max_seconds: float | None) -> Path:
    """Copy an HLS/DASH stream with ffmpeg, without re-encoding."""
    args = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-user_agent",
        "Mozilla/5.0 Dubloom/1.0",
        "-i",
        url,
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
    ]
    if max_seconds:
        args += ["-t", f"{max_seconds + 1:.0f}"]
    args += ["-c", "copy", "-bsf:a", "aac_adtstoasc", "-movflags", "+faststart", str(destination)]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode or not destination.is_file() or destination.stat().st_size == 0:
        raise SourceError("ffmpeg could not read this stream: " + result.stderr[-600:])
    return destination


def youtube_format(cap: int) -> str:
    """HLS picture and sound first (YouTube's direct DASH URLs can 403), the original language, the highest resolution
    within the cap. 'Premium' streams are skipped: YouTube resets their fragments for ordinary clients and the file comes
    back without picture. `!*=?` also keeps formats that carry no note at all, which is every HLS stream.
    """
    no_premium = "[format_note!*=?Premium]"
    return (
        f"bv*[protocol=m3u8_native][height<={cap}]{no_premium}+ba[protocol=m3u8_native]/"
        f"bv*[height<={cap}]{no_premium}+ba/b[height<={cap}]/bv*{no_premium}+ba/b"
    )


def _download_with_ytdlp(
    url: str,
    directory: Path,
    max_seconds: float | None,
    progress=None,
    section: tuple[float, float] | None = None,
) -> tuple[Path, str, dict, bool]:
    """Returns (path, title, meta, sectioned). With `section`, only that range is fetched (yt-dlp download ranges,
    keyframes forced at the cuts); a site that refuses ranges falls back to the whole video."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError, download_range_func

    hostname = (urlparse(url).hostname or "").lower()
    is_youtube = hostname in {"youtube.com", "youtu.be"} or hostname.endswith(".youtube.com")
    template = str(directory / "download.%(ext)s")
    cap = settings.max_video_height  # never fetch more than prepare_source keeps

    def hook(status):
        if not progress or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        done = status.get("downloaded_bytes") or 0
        if total:
            progress(min(0.99, done / total))

    options = {
        "format": f"bv*[height<={cap}]+ba/b[height<={cap}]/bv*+ba/b",
        "outtmpl": template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "restrictfilenames": True,
        "progress_hooks": [hook],
        # YouTube's throttling challenge needs Node >= 22 (or Deno); YTDLP_NODE points at one when PATH has an older node
        "js_runtimes": {
            "node": {"path": os.getenv("YTDLP_NODE")} if os.getenv("YTDLP_NODE") else {}
        },
        "retries": 3,
        "fragment_retries": 5,
        "socket_timeout": 30,
    }
    cookies = os.getenv("YTDLP_COOKIES", "").strip()
    if cookies and Path(cookies).is_file():
        # A Netscape cookies.txt from a signed-in browser clears YouTube's "confirm you're not a bot" check.
        options["cookiefile"] = cookies
    if is_youtube:
        # YouTube's direct media URLs can return 403 even for public videos, so prefer its fragment-based HLS
        # streams for both picture and sound. Among them: original language first, then the highest resolution
        # within the cap, then the highest bitrate. "Premium" (enhanced-bitrate) streams are skipped: their fragments
        # are reset by YouTube for non-premium clients and ffmpeg then writes a file with sound but no picture.
        options["format"] = youtube_format(cap)
        options["format_sort"] = ["lang", "res", "br"]
    sectioned = False
    if section:
        # only the part to dub: minutes of download become seconds on a long broadcast
        options["download_ranges"] = download_range_func(
            None, [(float(section[0]), float(section[1]))]
        )
        options["force_keyframes_at_cuts"] = True
        sectioned = True
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("is_live"):
                raise SourceError("Live streams are not supported")
            duration = info.get("duration")
            if section and duration and float(section[0]) >= duration:
                raise SourceError(
                    f"The section starts after the video ends ({int(duration // 60)}:{int(duration % 60):02d})"
                )
            # the limit is on what gets dubbed: the section when one was asked for, otherwise the whole video
            length = float(section[1]) - float(section[0]) if section else duration
            if max_seconds and length and length > max_seconds:
                raise SourceError(f"Video exceeds the {int(max_seconds // 60)}-minute limit")
            try:
                ydl.download([url])
            except DownloadError:
                if not sectioned:
                    raise
                if max_seconds and duration and duration > max_seconds:
                    raise SourceError(
                        f"This site only sends the whole video, which is longer than {int(max_seconds // 3600)} h; "
                        "download the part yourself and upload it"
                    )
                logger.warning("Range download refused for %s; fetching the whole video", url)
                sectioned = False
                for stale in directory.glob("download.*"):
                    stale.unlink(missing_ok=True)
                options.pop("download_ranges", None)
                options.pop("force_keyframes_at_cuts", None)
                with YoutubeDL(options) as whole:
                    whole.download([url])
            title = (info.get("title") or "").strip()
            meta = {
                "categories": info.get("categories") or [],
                "tags": (info.get("tags") or [])[:20],
                "channel": info.get("channel") or info.get("uploader") or "",
                "description": (info.get("description") or "")[:600],
            }
    except DownloadError as exc:
        message = str(exc).splitlines()[-1][:300]
        if "not a bot" in message or "Sign in to confirm" in message:
            message = (
                "YouTube is asking this server to sign in (bot check). Upload the video file instead, "
                "or set YTDLP_COOKIES to a cookies.txt exported from a signed-in browser."
            )
        raise SourceError(f"yt-dlp could not download this URL: {message}") from exc
    matches = sorted(
        p for p in directory.glob("download.*") if p.suffix.lower() in MEDIA_EXTENSIONS
    )
    if not matches:
        raise SourceError("The URL did not provide a downloadable video")
    return matches[0], title, meta, sectioned


def download_video(
    url: str,
    directory: Path,
    max_duration: float | None = None,
    max_bytes: int = 2048 * 1024 * 1024,
    progress=None,
    section: tuple[float, float] | None = None,
) -> SourceInfo:
    """Download from a direct media link, an HLS stream, or any yt-dlp supported page."""
    url = validate_url(url)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("download.*"):
        stale.unlink(missing_ok=True)
    content_type, _, final_url = _inspect_url(url)
    name = Path(unquote(urlparse(final_url).path)).stem or "video"
    if _is_direct_media(final_url, content_type):
        destination = directory / f"download{_guess_extension(final_url, content_type)}"
        return SourceInfo(
            _download_direct(final_url, destination, max_bytes, progress), name, "direct"
        )
    if content_type.split(";")[0].strip() in HLS_TYPES or final_url.lower().split("?")[0].endswith(
        (".m3u8", ".mpd")
    ):
        return SourceInfo(
            _download_stream(final_url, directory / "download.mp4", max_duration), name, "stream"
        )
    try:
        path, title, meta, sectioned = _download_with_ytdlp(
            url, directory, max_duration, progress, section
        )
        return SourceInfo(path, title or name, "yt-dlp", meta, sectioned)
    except SourceError:
        if Path(unquote(urlparse(final_url).path)).suffix.lower() in MEDIA_EXTENSIONS:
            destination = directory / f"download{_guess_extension(final_url, content_type)}"
            return SourceInfo(
                _download_direct(final_url, destination, max_bytes, progress), name, "direct"
            )
        raise


def prepare_source(raw: Path, output: Path, max_height: int = 720, progress=None) -> dict:
    """Produce a browser-playable H.264/AAC MP4 no taller than `max_height`."""
    info = probe(raw)
    if not info["has_video"]:
        raise SourceError("The file has no video stream; lip sync needs a picture")
    if not info["has_audio"]:
        raise SourceError("The video has no audio track to dub")
    if info["duration"] <= 0.5:
        raise SourceError("The video is too short to dub")
    container_ok = any(name in info["format_name"] for name in ("mp4", "mov"))
    codecs_ok = (
        info["video_codec"] == "h264"
        and info["audio_codec"] == "aac"
        and info["pix_fmt"] == "yuv420p"
    )
    size_ok = info["height"] <= max_height
    if container_ok and codecs_ok and size_ok:
        if raw.resolve() != output.resolve():
            shutil.move(str(raw), str(output))
        return probe(output)
    args = ["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-map", "0:v:0", "-map", "0:a:0"]
    if info["height"] > max_height:
        args += ["-vf", f"scale=-2:{max_height}"]
    else:
        args += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    args += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(output),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode or not output.is_file():
        raise SourceError("The video could not be converted: " + result.stderr[-600:])
    if raw.resolve() != output.resolve():
        raw.unlink(missing_ok=True)
    return probe(output)


PREVIEW_HEIGHT = 1080  # the in-page player streams this; the full-size master stays for download


def write_preview(master: Path, output: Path, max_height: int = PREVIEW_HEIGHT) -> Path | None:
    """A lighter H.264 rendition of `master` for smooth in-browser playback, or None when the master is
    already small enough (ultra-HD exports need ~15 Mbit/s, more than many viewers' connections sustain).
    """
    info = probe(master)
    if (
        min(info["width"], info["height"]) <= max_height
    ):  # 1080p landscape or 1080x1920 portrait plays as is
        output.unlink(missing_ok=True)
        return None
    scale = (
        f"scale=-2:{max_height}" if info["height"] <= info["width"] else f"scale={max_height}:-2"
    )
    run(
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        master,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-vf",
        scale,
        "-c:v",
        "libx264",
        "-preset",
        "faster",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        output,
    )
    return output


def make_poster(video: Path, output: Path, at_seconds: float = 1.0):
    """Write a 480 px wide still of `video` at `at_seconds`, or of its first frame if that fails."""
    try:
        run(
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{at_seconds:.2f}",
            "-i",
            video,
            "-frames:v",
            "1",
            "-vf",
            "scale=480:-2",
            "-q:v",
            "4",
            output,
        )
    except RuntimeError:
        run(
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            video,
            "-frames:v",
            "1",
            "-vf",
            "scale=480:-2",
            "-q:v",
            "4",
            output,
        )


def extract_audio(video: Path, output: Path):
    """Extract the source soundtrack as mono, 16 kHz, 16-bit PCM WAV."""
    run(
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        video,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        output,
    )


def cut_audio(audio: Path, start: float, end: float, output: Path):
    """Write a time-bounded mono, 16 kHz, 16-bit PCM WAV for model input."""
    run(
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{end - start:.3f}",
        "-i",
        audio,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        output,
    )
