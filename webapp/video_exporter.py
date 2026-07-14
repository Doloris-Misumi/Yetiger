from __future__ import annotations

import io
import math
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    from audio_assets import CALL_AUDIO_FILES as CALL_AUDIO_MAP
except ImportError:  # pragma: no cover - package-style imports in tests/tools.
    from webapp.audio_assets import CALL_AUDIO_FILES as CALL_AUDIO_MAP


ROOT = Path(__file__).resolve().parent.parent

WIDTH = 1280
HEIGHT = 720
GEI_VIDEO_MAP = {
    "long_zhi_mao": ROOT / "gei_video" / "龙之矛.mp4",
    "lei_she": ROOT / "gei_video" / "雷蛇.mp4",
}

ROLE_COLORS = {
    "keepspace": "#6b7280",
    "rhythmcall": "#1d8f74",
    "mix": "#c65347",
    "underground_gei": "#7a4fa3",
}
MUSIC_COLORS = {
    "intro": "#2f6fb2",
    "verse": "#1d8f74",
    "pre_chorus": "#b7791f",
    "pre_chorus_build": "#d97706",
    "chorus": "#c65347",
    "post_chorus": "#9f5f2a",
    "bridge": "#7a4fa3",
    "instrumental": "#0f766e",
    "instrumental_break": "#0f766e",
    "interlude": "#0f766e",
    "solo": "#8b5cf6",
    "outro": "#475467",
    "end": "#334155",
    "unknown": "#64748b",
}
ROLE_LABELS = {
    "keepspace": "留白",
    "rhythmcall": "call",
    "mix": "MIX",
    "underground_gei": "地下艺",
}
RISK_LABELS = {
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
}


def resolve_ffmpeg() -> str:
    candidates = [
        ROOT / ".venv" / "Scripts" / "ffmpeg.exe",
        ROOT / ".venv" / "bin" / "ffmpeg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    resolved = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if resolved:
        return resolved
    raise FileNotFoundError("ffmpeg executable not found in .venv or PATH")


def load_gei_frames(video_path: Path, fps: int) -> Tuple[List[Image.Image], float]:
    """Extract all frames from a gei video at given fps. Returns (frames, duration)."""
    tmp = tempfile.mkdtemp(prefix="yetiger_gei_")
    try:
        cmd = [
            resolve_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", f"fps={fps},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2",
            f"{tmp}/frame_%04d.png",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        frame_files = sorted(Path(tmp).glob("frame_*.png"))
        frames = [Image.open(str(p)).convert("RGB") for p in frame_files]
        if not frames:
            return [], 0.0
        duration = len(frames) / fps
        return frames, duration
    finally:
        for p in Path(tmp).glob("frame_*.png"):
            try: p.unlink()
            except OSError: pass
        try: Path(tmp).rmdir()
        except OSError: pass


def build_gei_cache(result: Dict[str, Any], fps: int) -> Dict[str, Tuple[List[Image.Image], float]]:
    """Pre-load gei video frames for all actions referenced in the timeline."""
    cache: Dict[str, Tuple[List[Image.Image], float]] = {}
    seen: set = set()
    for item in result.get("timeline") or []:
        action_id = str(item.get("action_id") or "")
        if not action_id or action_id in seen:
            continue
        video_path = GEI_VIDEO_MAP.get(action_id)
        if video_path and video_path.exists():
            seen.add(action_id)
            try:
                frames, duration = load_gei_frames(video_path, fps)
                if frames:
                    cache[action_id] = (frames, duration)
            except Exception:
                pass
    return cache


def build_call_audio_track(result: Dict[str, Any], duration: float, tmp_dir: Path) -> Optional[Path]:
    """Generate a mixed call-audio WAV for the full song duration, or None if no calls have audio."""
    timeline = result.get("timeline") or []
    call_entries = []
    for item in timeline:
        action_id = str(item.get("action_id") or "")
        audio_path = CALL_AUDIO_MAP.get(action_id)
        if audio_path and audio_path.exists():
            call_entries.append((item, audio_path))
    if not call_entries:
        return None

    ffmpeg = resolve_ffmpeg()
    mixed_path = tmp_dir / "call_mix.wav"

    filter_parts = []
    for idx, (item, audio_path) in enumerate(call_entries):
        action_start = float(item.get("start") or 0)
        action_end = float(item.get("end") or action_start + 1)
        action_duration = max(0.1, action_end - action_start)
        filter_parts.append(
            f"[{idx + 1}:a]atempo={min(4.0, max(0.5, 1.0))}[s{idx}];"
            f"[s{idx}]adelay={int(action_start * 1000)}|{int(action_start * 1000)}:all=1[a{idx}]"
        )

    inputs = []
    for _, audio_path in call_entries:
        inputs.extend(["-i", str(audio_path)])

    amix_inputs = "".join(f"[a{i}]" for i in range(len(call_entries)))
    filter_complex = ";".join(
        f"[{i + 1}:a]atempo={min(4.0, max(0.5, 1.0))}[s{i}];"
        f"[s{i}]adelay={int(float(call_entries[i][0].get('start') or 0) * 1000)}|{int(float(call_entries[i][0].get('start') or 0) * 1000)}:all=1[a{i}]"
        for i in range(len(call_entries))
    ) + f";{amix_inputs}amix=inputs={len(call_entries)}:duration=first:dropout_transition=0,volume={min(1.0, 0.8 / max(1, len(call_entries)))}[aout]"

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-t", f"{duration:.3f}",
        str(mixed_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return mixed_path if mixed_path.exists() else None


def fmt_time(seconds: float) -> str:
    safe = max(0.0, float(seconds or 0.0))
    minutes = int(safe // 60)
    secs = safe - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    clean = str(value or "#000000").lstrip("#")
    if len(clean) != 6:
        return 0, 0, 0
    return tuple(int(clean[index : index + 2], 16) for index in (0, 2, 4))


def blend(color: str, alpha: float, base: str = "#000000") -> Tuple[int, int, int]:
    fg = hex_to_rgb(color)
    bg = hex_to_rgb(base)
    return tuple(round(fg[i] * alpha + bg[i] * (1.0 - alpha)) for i in range(3))


@lru_cache(maxsize=128)
def load_font(size: int, *, bold: bool = False, serif: bool = False) -> ImageFont.ImageFont:
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = []
    if serif:
        candidates.extend([
            windows / "simsunb.ttf",
            windows / "simsun.ttc",
        ])
    if bold:
        candidates.extend([
            windows / "msyhbd.ttc",
            windows / "arialbd.ttf",
        ])
    candidates.extend([
        windows / "msyh.ttc",
        windows / "simsun.ttc",
        windows / "arial.ttf",
    ])
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), str(text), font=font)
    return int(box[2] - box[0])


def text_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), str(text), font=font)
    return int(box[3] - box[1])


def tokenise(text: str) -> List[str]:
    pattern = r"[\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]|[^\s\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]+"
    return re.findall(pattern, str(text or "").replace("\n", " "))


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: Optional[int] = None,
) -> List[str]:
    lines: List[str] = []
    line = ""
    for token in tokenise(text):
        glue = " " if line and not re.match(r"^[\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]$", token) else ""
        candidate = f"{line}{glue}{token}"
        if text_width(draw, candidate, font) <= max_width or not line:
            line = candidate
        else:
            lines.append(line)
            line = token
    if line:
        lines.append(line)
    if max_lines and len(lines) > max_lines:
        clipped = lines[:max_lines]
        clipped[-1] = clipped[-1].rstrip(". ") + "..."
        return clipped
    return lines


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    size: int,
    *,
    min_size: int = 20,
    bold: bool = False,
    serif: bool = False,
) -> ImageFont.ImageFont:
    current = size
    while current >= min_size:
        font = load_font(current, bold=bold, serif=serif)
        if text_width(draw, text, font) <= max_width:
            return font
        current -= 2
    return load_font(min_size, bold=bold, serif=serif)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: Tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_height: int,
    max_lines: int,
) -> int:
    lines = wrap_text(draw, text, font, max_width, max_lines)
    x, y = xy
    for index, line in enumerate(lines):
        draw.text((x, y + index * line_height), line, font=font, fill=fill)
    return len(lines)


def draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    center_x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: str,
    line_height: int,
) -> None:
    for index, line in enumerate(lines):
        width = text_width(draw, line, font)
        draw.text((center_x - width // 2, y + index * line_height), line, font=font, fill=fill)


def item_start(item: Dict[str, Any]) -> float:
    return float(item.get("start") or 0.0)


def item_end(item: Dict[str, Any]) -> float:
    return float(item.get("end") or item_start(item))


def item_at(time_s: float, items: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in items:
        if item_start(item) <= time_s < item_end(item):
            return item
    return None


def next_item(time_s: float, items: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    future = [item for item in items if item_start(item) > time_s]
    if not future:
        return None
    return min(future, key=item_start)


def current_tutorial_cue(action: Optional[Dict[str, Any]], time_s: float) -> Optional[Dict[str, Any]]:
    if not action:
        return None
    tutorial = action.get("tutorial_text") or {}
    bars = [str(item).strip() for item in tutorial.get("bars") or [] if str(item).strip()]
    if not bars:
        return None
    start = item_start(action)
    end = max(start + 0.001, item_end(action))
    progress = clamp((time_s - start) / (end - start), 0.0, 0.999999)
    index = min(len(bars) - 1, int(progress * len(bars)))
    return {
        "index": index,
        "total": len(bars),
        "text": bars[index],
        "source": tutorial.get("source"),
    }


def draw_label(draw: ImageDraw.ImageDraw, panel: Tuple[int, int, int, int], label: str) -> None:
    x, y, _, _ = panel
    draw.text((x + 26, y + 34), label, font=load_font(18, bold=True), fill="#e5e7eb")


def draw_note_panel(
    draw: ImageDraw.ImageDraw,
    panel: Tuple[int, int, int, int],
    result: Dict[str, Any],
    current: Optional[Dict[str, Any]],
    upcoming: Optional[Dict[str, Any]],
    music: Optional[Dict[str, Any]],
    duration: float,
    time_s: float,
    role_color: str,
) -> None:
    x, y, width, height = panel
    inner_x = x + 30
    max_width = width - 60
    draw_label(draw, panel, "备注")

    title = str((result.get("song") or {}).get("title") or "YesTiger")
    title_font = fit_font(draw, title, max_width, 40, min_size=24, bold=True)
    draw_wrapped(draw, title, (inner_x, y + 84), title_font, "#f8fafc", max_width, 44, 2)

    section = (music or current or {}).get("music_label") or "-"
    window = f"{fmt_time(item_start(current))}-{fmt_time(item_end(current))}" if current else f"Now {fmt_time(time_s)}"
    rows = [
        f"当前时间  {fmt_time(time_s)} / {fmt_time(duration)}",
        f"段落  {section}",
        f"动作区间  {window}",
        f"小节  {(current or {}).get('bar_count', '-')} bars",
        f"风险  {RISK_LABELS.get((current or {}).get('risk'), (current or {}).get('risk') or '低风险')}",
    ]
    row_font = load_font(24)
    for index, row in enumerate(rows):
        draw.text((inner_x, y + 192 + index * 36), row, font=row_font, fill="#d1d5db")

    progress_width = int(max_width * clamp(time_s / max(0.001, duration), 0.0, 1.0))
    progress_y = y + height - 78
    draw.rectangle((inner_x, progress_y, inner_x + max_width, progress_y + 8), outline="#f8fafc", width=2)
    draw.rectangle((inner_x, progress_y, inner_x + progress_width, progress_y + 8), fill=role_color)

    if upcoming:
        next_text = f"Next  {fmt_time(item_start(upcoming))}  {upcoming.get('display_name') or '-'}"
        draw_wrapped(draw, next_text, (inner_x, y + height - 42), load_font(20), "#9ca3af", max_width, 24, 1)


def draw_media_panel(
    draw: ImageDraw.ImageDraw,
    panel: Tuple[int, int, int, int],
    result: Dict[str, Any],
    current: Optional[Dict[str, Any]],
    music: Optional[Dict[str, Any]],
    role_color: str,
    time_s: float = 0.0,
    gei_cache: Optional[Dict[str, Tuple[List[Image.Image], float]]] = None,
) -> None:
    x, y, width, height = panel

    action_id = str((current or {}).get("action_id") or "")
    if gei_cache and action_id in gei_cache:
        frames, video_duration = gei_cache[action_id]
        if frames:
            action_start = float((current or {}).get("start") or 0)
            action_end = float((current or {}).get("end") or action_start + 1)
            action_duration = max(0.001, action_end - action_start)
            local_time = max(0.0, min(action_duration, time_s - action_start))
            progress = local_time / action_duration
            frame_idx = min(len(frames) - 1, int(progress * len(frames)))
            frame_img = frames[frame_idx].resize((width, height), Image.LANCZOS)
            draw._image.paste(frame_img, (x, y))
            draw.rectangle((x, y + height - 6, x + width, y + height), fill=role_color)
            title = f"{(current or {}).get('display_name') or ''} · 演示动作"
            title_font = fit_font(draw, title, width - 80, 32, min_size=18, bold=True)
            draw_centered_lines(draw, [title], x + width // 2, y + 42, title_font, "#ffffff", 36)
            draw_centered_lines(draw, ["视频同步播放中"], x + width // 2, y + height - 36, load_font(20), "#ffffff", 24)
            return

    accent = MUSIC_COLORS.get((music or {}).get("music_label"), role_color)
    draw.rectangle((x, y, x + width, y + height), fill=blend(accent, 0.9))
    for col in range(16):
        size = 14 + (col % 3) * 8
        px = x + 40 + col * 58
        py = y + 56 + (col % 5) * 52
        draw.rectangle((px, py, px + size, py + size), fill=blend("#ffffff", 0.18, accent))
    draw.rectangle((x, y + height - 108, x + width, y + height), fill=blend("#000000", 0.18, accent))

    title_font = fit_font(draw, "MV / DEMO SLOT", width - 100, 66, min_size=32, bold=True)
    draw_centered_lines(
        draw,
        ["MV / DEMO SLOT"],
        x + width // 2,
        y + height // 2 - 48,
        title_font,
        "#ffffff",
        70,
    )
    placeholder = "后续接入地下艺演示动作" if (current or {}).get("role") == "underground_gei" else "后续接入该歌曲 MV 或教学画面"
    draw_centered_lines(
        draw,
        [placeholder],
        x + width // 2,
        y + height // 2 + 30,
        load_font(28, bold=True),
        "#ffffff",
        36,
    )
    caption = f"{(result.get('song') or {}).get('title') or 'YesTiger'} · {(music or current or {}).get('music_label') or '-'}"
    caption_font = fit_font(draw, caption, width - 80, 24, min_size=18)
    draw_centered_lines(draw, [caption], x + width // 2, y + height - 48, caption_font, "#ffffff", 28)


def draw_action_panel(
    draw: ImageDraw.ImageDraw,
    panel: Tuple[int, int, int, int],
    current: Optional[Dict[str, Any]],
    role_color: str,
) -> None:
    x, y, width, height = panel
    inner_x = x + 30
    max_width = width - 60
    draw_label(draw, panel, "应援种类及名称")
    draw.rectangle((inner_x, y + 76, inner_x + max_width, y + 84), fill=role_color)

    role_text = ROLE_LABELS.get((current or {}).get("role"), ROLE_LABELS["keepspace"])
    role_font = fit_font(draw, role_text, max_width, 48, min_size=26, bold=True)
    draw.text((inner_x, y + 112), role_text, font=role_font, fill="#f8fafc")

    action_name = str((current or {}).get("display_name") or "Keep Space")
    action_font = load_font(40, bold=True)
    draw_wrapped(draw, action_name, (inner_x, y + 194), action_font, "#f8fafc", max_width, 46, 2)

    if current:
        meta = (
            f"{current.get('music_label') or '-'} · {current.get('bar_count', '-')} bars · "
            f"{RISK_LABELS.get(current.get('risk'), current.get('risk') or 'low')}"
        )
    else:
        meta = "No action loaded"
    draw_wrapped(draw, meta, (inner_x, y + height - 48), load_font(22), "#a8b3c2", max_width, 26, 1)


def draw_method_panel(
    draw: ImageDraw.ImageDraw,
    panel: Tuple[int, int, int, int],
    current: Optional[Dict[str, Any]],
    time_s: float,
    duration: float,
    role_color: str,
) -> None:
    x, y, width, height = panel
    inner_x = x + 54
    max_width = width - 108
    draw_label(draw, panel, "具体打法")

    cue = current_tutorial_cue(current, time_s)
    if cue:
        source = f" · {cue['source']}" if cue.get("source") else ""
        cue_label = f"Bar cue {cue['index'] + 1}/{cue['total']}{source}"
        draw.text((inner_x, y + 70), cue_label, font=load_font(22, bold=True), fill="#a8b3c2")
    text = (
        (cue or {}).get("text")
        or (current or {}).get("typical_text")
        or (f"{(current or {}).get('display_name') or 'Action'}：按当前段落节拍执行。" if current else "")
    )
    font = fit_font(draw, str(text), max_width, 44, min_size=28, bold=True, serif=False)
    line_height = max(36, text_height(draw, "Ag", font) + 18)
    lines = wrap_text(draw, str(text), font, max_width, 4)
    start_y = y + 112 + max(0, (height - 184 - len(lines) * line_height) // 2)
    draw_centered_lines(draw, lines, x + width // 2, start_y, font, "#f8fafc", line_height)

    progress_width = int(max_width * clamp(time_s / max(0.001, duration), 0.0, 1.0))
    progress_y = y + height - 42
    draw.rectangle((inner_x, progress_y, inner_x + max_width, progress_y + 10), fill="#1f2937")
    draw.rectangle((inner_x, progress_y, inner_x + progress_width, progress_y + 10), fill=role_color)


def render_frame(result: Dict[str, Any], time_s: float, duration: float, gei_cache=None) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)
    gap = 6
    left_width = 386
    top_height = 430
    right_width = WIDTH - left_width - gap
    bottom_height = HEIGHT - top_height - gap
    panels = {
        "note": (0, 0, left_width, top_height),
        "media": (left_width + gap, 0, right_width, top_height),
        "action": (0, top_height + gap, left_width, bottom_height),
        "method": (left_width + gap, top_height + gap, right_width, bottom_height),
    }
    for x, y, width, height in panels.values():
        draw.rectangle((x, y, x + width, y + height), fill="#050505")

    timeline = result.get("timeline") or []
    music_segments = result.get("music_segments") or result.get("segments") or []
    current = item_at(time_s, timeline)
    upcoming = next_item(time_s, timeline)
    music = item_at(time_s, music_segments)
    role = (current or {}).get("role") or "keepspace"
    role_color = ROLE_COLORS.get(role, ROLE_COLORS["keepspace"])

    draw_note_panel(draw, panels["note"], result, current, upcoming, music, duration, time_s, role_color)
    draw_media_panel(draw, panels["media"], result, current, music, role_color, time_s=time_s, gei_cache=gei_cache)
    draw_action_panel(draw, panels["action"], current, role_color)
    draw_method_panel(draw, panels["method"], current, time_s, duration, role_color)
    return image


def export_teaching_video(
    result: Dict[str, Any],
    audio_path: Path,
    output_path: Path,
    *,
    fps: Optional[int] = None,
) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    ffmpeg = resolve_ffmpeg()
    fps = int(fps or os.environ.get("YESTIGER_EXPORT_FPS", "10"))
    fps = max(4, min(30, fps))
    song = result.get("song") or {}
    duration = float(song.get("duration") or 0.0)
    if duration <= 0:
        duration = max((item_end(item) for item in result.get("timeline") or []), default=0.0)
    if duration <= 0:
        raise ValueError("Cannot export video without a positive song duration")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(math.ceil(duration * fps)))

    tmp_dir = Path(tempfile.mkdtemp(prefix="yetiger_export_"))
    call_track = None
    try:
        call_track = build_call_audio_track(result, duration, tmp_dir)
    except Exception:
        pass

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(audio_path),
    ]
    if call_track and call_track.exists():
        command.extend(["-i", str(call_track)])

    command.extend([
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
    ])

    if call_track and call_track.exists():
        command.extend([
            "-filter_complex",
            "[1:a][2:a]amix=inputs=2:duration=first:weights=1 1[aout]",
            "-map",
            "[aout]",
        ])
    else:
        command.extend(["-map", "1:a:0"])

    command.extend([
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ])
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    gei_cache = build_gei_cache(result, fps)
    try:
        for index in range(frame_count):
            time_s = min(duration - 0.001, index / fps)
            frame = render_frame(result, time_s, duration, gei_cache=gei_cache)
            process.stdin.write(frame.tobytes())
    except BrokenPipeError as exc:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise RuntimeError(f"ffmpeg stopped while receiving frames: {stderr}") from exc
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            for p in tmp_dir.glob("*"):
                p.unlink()
            tmp_dir.rmdir()
        except OSError:
            pass

    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait(timeout=max(60, int(duration * 4)))
    if return_code != 0:
        raise RuntimeError(stderr[-2000:] or f"ffmpeg exited with {return_code}")
    return output_path
