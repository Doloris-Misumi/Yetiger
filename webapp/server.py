import argparse
import cgi
from email.header import decode_header, make_header
import json
import mimetypes
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

# Force torchaudio to use soundfile backend (avoids torchcodec + FFmpeg dep)
os.environ.setdefault("TORCHAUDIO_USE_SOUNDFILE_LEGACY_INTERFACE", "1")

# Redirect stderr to stdout so Render captures ALL errors
sys.stderr = sys.stdout

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# ── Lightweight helpers (inlined to avoid heavy analyzer.py import chain) ──

import re as _re

ROOT = THIS_DIR.parent
RUN_DIR = Path(os.environ.get("YESTIGER_RUN_DIR") or (ROOT / "webapp_runs")).resolve()
UPLOAD_DIR = RUN_DIR / "uploads"
JOB_DIR = RUN_DIR / "jobs"


def _slugify(value):
    cleaned = _re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(value).strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "uploaded_song"


slugify = _slugify


# Heavy analyzer imports — deferred to first use
_analyzer = None
_analyzer_error = None
_analyzer_ready = threading.Event()


def _get_analyzer():
    """Lazy-load analyzer module with background pre-warm."""
    global _analyzer
    _analyzer_ready.wait()  # Wait for background import to finish
    if _analyzer is None:
        raise RuntimeError(f"Analyzer failed to load: {_analyzer_error}")
    return _analyzer


def _prewarm_analyzer():
    global _analyzer, _analyzer_error
    try:
        import analyzer as _mod
        _analyzer = _mod
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _analyzer_error = exc
    finally:
        _analyzer_ready.set()


from audio_assets import call_audio_path


def _get_video_exporter():
    from video_exporter import export_teaching_video as _fn
    return _fn


STATIC_DIR = THIS_DIR / "static"
EXAMPLE_AUDIO_DIR = Path(os.environ.get("YESTIGER_EXAMPLE_AUDIO_DIR") or (ROOT / "example_audio")).resolve()
EXAMPLE_AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

# In-memory job status for async analysis (bypasses Render 30s timeout)
_job_status = {}
_job_status_lock = threading.Lock()


def _set_job_status(job_id: str, status: str, result=None):
    with _job_status_lock:
        entry = _job_status.get(job_id, {})
        entry["status"] = status
        if result is not None:
            entry["result"] = result
        _job_status[job_id] = entry


def _get_job_status(job_id: str):
    with _job_status_lock:
        return _job_status.get(job_id, {"status": "not_found"})


def _run_analysis_async(audio_path, title, job_id, original_filename):
    try:
        print(f"[job {job_id}] Analysis thread started", flush=True)
        _set_job_status(job_id, "processing")
        print(f"[job {job_id}] Calling analyze_audio...", flush=True)
        result = _get_analyzer().analyze_audio(audio_path, title=title, job_id=job_id)
        print(f"[job {job_id}] analyze_audio done", flush=True)
        result.setdefault("song", {})["original_audio_filename"] = original_filename
        result["audio_url"] = f"/api/jobs/{job_id}/audio"
        result["downloads"] = {
            "json": f"/api/jobs/{job_id}/result.json",
            "markdown": f"/api/jobs/{job_id}/callbook.md",
        }
        _get_analyzer().save_analysis_result(result, JOB_DIR / job_id)
        print(f"[job {job_id}] Result saved", flush=True)
        _set_job_status(job_id, "done", result)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[job {job_id}] ERROR: {exc}", flush=True)
        _set_job_status(job_id, "error", {"error": str(exc)})


def default_port() -> int:
    try:
        return int(os.environ.get("PORT", "8765"))
    except ValueError:
        return 8765


def cors_origin(origin: str) -> str:
    configured = os.environ.get("YESTIGER_CORS_ORIGIN", "*").strip()
    if not configured or configured == "*":
        return "*"
    origin = str(origin or "").rstrip("/")
    allowed = [item.strip().rstrip("/") for item in configured.split(",") if item.strip()]
    return origin if origin in allowed else ""


def safe_join(base: Path, raw_path: str) -> Path:
    requested = (base / raw_path.lstrip("/")).resolve()
    if base.resolve() not in requested.parents and requested != base.resolve():
        raise ValueError("Path escapes static root.")
    return requested


def guess_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def example_audio_path(song_id: str) -> Path:
    safe_song_id = slugify(song_id)
    for suffix in EXAMPLE_AUDIO_SUFFIXES:
        candidate = EXAMPLE_AUDIO_DIR / f"{safe_song_id}{suffix}"
        if candidate.exists():
            return candidate

    annotation_path = ROOT / "train" / "annotations" / safe_song_id / f"{safe_song_id}.annotation.json"
    raw = None
    if annotation_path.exists():
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        raw = (data.get("song") or {}).get("audio_path")
    path = Path(str(raw)) if raw else ROOT / "train" / "songs" / f"{safe_song_id}.mp3"
    audio = path if path.is_absolute() else ROOT / path
    if not audio.exists():
        raise FileNotFoundError(audio)
    return audio


def uploaded_job_audio_path(job_id: str) -> Path:
    matches = list((UPLOAD_DIR / slugify(job_id)).glob("*"))
    if not matches:
        raise FileNotFoundError(f"audio not found for job {job_id}")
    return matches[0]


def resolve_result_audio_path(result: dict) -> Path:
    audio_url = str(result.get("audio_url") or "")
    if audio_url:
        parsed = urlparse(audio_url)
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) >= 3 and parts[:2] == ["api", "example-audio"]:
            return example_audio_path(slugify(parts[2]))
        if len(parts) >= 4 and parts[:2] == ["api", "jobs"] and parts[3] == "audio":
            return uploaded_job_audio_path(parts[2])

    raw_audio_path = result.get("audio_path")
    if raw_audio_path:
        candidate = Path(str(raw_audio_path))
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        root = ROOT.resolve()
        if candidate == root or root in candidate.parents:
            if candidate.exists():
                return candidate

    job_id = str(result.get("job_id") or "")
    if job_id.startswith("example_"):
        return example_audio_path(_slugify(job_id.replace("example_", "", 1)))
    if job_id:
        try:
            return uploaded_job_audio_path(job_id)
        except FileNotFoundError:
            pass

    song_id = slugify(((result.get("song") or {}).get("song_id")) or "")
    if song_id:
        candidate = _get_analyzer().TRAIN_DIR / "songs" / f"{song_id}.mp3"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve audio for video export")


def export_job_id_from_result(result: dict) -> str:
    audio_url = str(result.get("audio_url") or "")
    if audio_url:
        parts = [part for part in urlparse(audio_url).path.strip("/").split("/") if part]
        if len(parts) >= 3 and parts[:2] == ["api", "example-audio"]:
            return f"example_{slugify(parts[2])}"
        if len(parts) >= 4 and parts[:2] == ["api", "jobs"] and parts[3] == "audio":
            return slugify(parts[2])
    return slugify(result.get("job_id") or (result.get("song") or {}).get("song_id") or "video_export")


def decode_upload_filename(value: str) -> str:
    raw = str(value or "uploaded_audio.wav")
    try:
        decoded = str(make_header(decode_header(raw)))
    except Exception:
        decoded = raw
    return Path(decoded).name or "uploaded_audio.wav"


def suffix_from_content_type(content_type: str) -> str:
    content_type = str(content_type or "").split(";")[0].strip().lower()
    if content_type in {"audio/mpeg", "audio/mp3", "audio/x-mpeg"}:
        return ".mp3"
    if content_type in {"audio/wav", "audio/wave", "audio/x-wav"}:
        return ".wav"
    if content_type == "audio/flac":
        return ".flac"
    if content_type in {"audio/mp4", "audio/x-m4a"}:
        return ".m4a"
    if content_type in {"audio/ogg", "application/ogg"}:
        return ".ogg"
    return ".audio"


class YesTigerHandler(BaseHTTPRequestHandler):
    server_version = "YesTigerWeb/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def end_headers(self) -> None:
        origin = cors_origin(self.headers.get("Origin", ""))
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            if origin != "*":
                self.send_header("Vary", "Origin")
        super().end_headers()

    def send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str = "text/plain; charset=utf-8", status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, download_name: str = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_json({"error": "file_not_found"}, status=404)
            return
        file_size = path.stat().st_size
        content_type = guess_type(path)
        range_header = self.headers.get("Range", "")
        if range_header and range_header.startswith("bytes="):
            range_value = range_header[6:].strip()
            if "-" in range_value:
                parts = range_value.split("-", 1)
                range_start = int(parts[0]) if parts[0] else 0
                range_end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
            else:
                range_start = int(range_value)
                range_end = file_size - 1
            range_start = max(0, min(range_start, file_size - 1))
            range_end = max(range_start, min(range_end, file_size - 1))
            chunk_size = range_end - range_start + 1
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Range", f"bytes {range_start}-{range_end}/{file_size}")
            self.send_header("Content-Length", str(chunk_size))
            self.send_header("Accept-Ranges", "bytes")
            if download_name:
                self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(range_start)
                self.wfile.write(handle.read(chunk_size))
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/" or path == "/index.html":
                self.send_file(STATIC_DIR / "index.html")
                return
            if path == "/studio" or path == "/studio.html":
                self.send_file(STATIC_DIR / "studio.html")
                return
            if path == "/builder" or path == "/builder.html":
                self.send_file(STATIC_DIR / "builder.html")
                return
            if path in {"/readme", "/readme.html", "/help", "/help.html"}:
                self.send_file(STATIC_DIR / "readme.html")
                return
            if path in {"/styles.css", "/config.js", "/app.js", "/builder.js"}:
                self.send_file(STATIC_DIR / path.lstrip("/"))
                return
            if path.startswith("/static/"):
                self.send_file(safe_join(STATIC_DIR, path[len("/static/") :]))
                return
            if path.startswith("/examples/"):
                self.send_file(safe_join(STATIC_DIR / "examples", path[len("/examples/") :]))
                return
            if path == "/api/songs":
                self.handle_songs()
                return
            if path.startswith("/api/examples/"):
                song_id = slugify(path.split("/")[-1])
                self.handle_example(song_id)
                return
            if path.startswith("/api/example-audio/"):
                song_id = slugify(path.split("/")[-1])
                try:
                    self.send_file(example_audio_path(song_id))
                except FileNotFoundError:
                    self.send_json({
                        "error": "example_audio_not_found",
                        "message": (
                            f"Put a licensed audio file at example_audio/{song_id}.mp3 "
                            "or set YESTIGER_EXAMPLE_AUDIO_DIR."
                        ),
                    }, status=404)
                return
            if path.startswith("/api/jobs/"):
                if path.endswith("/status"):
                    job_id = slugify(path.split("/")[-2])
                    self.send_json(_get_job_status(job_id))
                    return
                self.handle_job_file(path)
                return
            if path == "/api/health":
                self.send_json({
                    "status": "ok" if _analyzer is not None else "loading" if not _analyzer_ready.is_set() else "error",
                    "ready": _analyzer_ready.is_set(),
                    "analyzer_loaded": _analyzer is not None,
                    "analyzer_error": str(_analyzer_error) if _analyzer_error else None,
                    "run_dir": str(RUN_DIR),
                })
                return
            if path == "/api/actions":
                if _analyzer_ready.is_set():
                    self.send_json({"actions": _get_analyzer().list_action_library()})
                else:
                    self.send_json({"actions": [], "status": "loading"})
                return
            if path == "/api/custom-actions":
                if _analyzer_ready.is_set():
                    self.send_json({"actions": _get_analyzer().list_custom_actions()})
                else:
                    self.send_json({"actions": [], "status": "loading"})
                return
            if path.startswith("/api/gei-video/"):
                video_name = unquote(path.split("/")[-1]).strip()
                gei_dir = ROOT / "gei_video"
                video_map = {
                    "long_zhi_mao": "龙之矛.mp4",
                    "lei_she": "雷蛇.mp4",
                }
                filename = video_map.get(video_name)
                if filename:
                    candidate = gei_dir / filename
                    if candidate.exists():
                        self.send_file(candidate)
                        return
                for candidate in gei_dir.glob(f"*{video_name}*"):
                    self.send_file(candidate)
                    return
                self.send_json({"error": "video_not_found"}, status=404)
                return
            if path.startswith("/api/call-audio/"):
                action_id = unquote(path.split("/")[-1]).strip()
                candidate = call_audio_path(action_id)
                if candidate and candidate.exists():
                    self.send_file(candidate)
                    return
                self.send_json({"error": "audio_not_found"}, status=404)
                return
            self.send_json({"error": "not_found"}, status=404)
        except Exception as exc:
            self.send_json({"error": type(exc).__name__, "message": str(exc)}, status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/analyze":
            try:
                self.handle_analyze()
            except Exception as exc:
                self.send_json({"error": type(exc).__name__, "message": str(exc)}, status=500)
            return
        if parsed.path == "/api/export-video":
            try:
                self.handle_export_video()
            except Exception as exc:
                self.send_json({"error": type(exc).__name__, "message": str(exc)}, status=500)
            return
        if parsed.path == "/api/custom-actions":
            try:
                self.handle_save_custom_action()
            except Exception as exc:
                self.send_json({"error": type(exc).__name__, "message": str(exc)}, status=500)
            return
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/save"):
            try:
                self.handle_save_job(parsed.path)
            except Exception as exc:
                self.send_json({"error": type(exc).__name__, "message": str(exc)}, status=500)
            return
        else:
            self.send_json({"error": "not_found"}, status=404)
            return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def handle_songs(self) -> None:
        self.send_json({"songs": _get_analyzer().list_example_songs()})

    def handle_example(self, song_id: str) -> None:
        result = _get_analyzer().load_example_result(song_id)
        if result.get("audio_path"):
            result["audio_url"] = f"/api/example-audio/{song_id}"
            result.pop("audio_path", None)
        result["downloads"] = {}
        self.send_json(result)

    def handle_job_file(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 4 or len(parts) > 5:
            self.send_json({"error": "bad_job_path"}, status=400)
            return
        _, _, job_id, filename = parts[:4]
        job_id = slugify(job_id)
        job_dir = JOB_DIR / job_id
        if len(parts) == 5 and parts[3] == "gei_clips":
            # /api/jobs/<job_id>/gei_clips/<clip_name>
            clip_name = _slugify(parts[4])
            gei_dir = job_dir / "gei_clips"
            clip_path = gei_dir / f"{clip_name}.mp4" if not clip_name.endswith(".mp4") else gei_dir / clip_name
            if not clip_path.exists():
                # Try glob match
                matches = list(gei_dir.glob(f"*{clip_name}*"))
                if matches:
                    clip_path = matches[0]
            self.send_file(clip_path)
            return
        if filename == "result.json":
            self.send_file(job_dir / "result.json", download_name=f"{job_id}.result.json")
            return
        if filename == "callbook.md":
            self.send_file(job_dir / "callbook.md", download_name=f"{job_id}.callbook.md")
            return
        if filename == "audio":
            matches = list((UPLOAD_DIR / job_id).glob("*"))
            if not matches:
                self.send_json({"error": "audio_not_found"}, status=404)
                return
            self.send_file(matches[0])
            return
        self.send_json({"error": "bad_job_file"}, status=400)

    def handle_analyze(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json({"error": "expected_multipart_form_data"}, status=400)
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )
        if "audio" not in form:
            self.send_json({"error": "missing_audio"}, status=400)
            return
        field = form["audio"]
        if isinstance(field, list):
            field = field[0]
        original_filename = decode_upload_filename(field.filename or "uploaded_audio.wav")
        original_path = Path(original_filename)
        suffix = original_path.suffix.lower()
        safe_suffix = suffix if suffix and len(suffix) <= 10 and all(char.isalnum() or char == "." for char in suffix) else suffix_from_content_type(getattr(field, "type", ""))
        safe_stem = _slugify(original_path.stem)[:80] or "uploaded_audio"
        filename = f"{safe_stem}{safe_suffix}"
        title = form.getfirst("title") or original_path.stem
        job_id = slugify(f"{safe_stem}_{_slugify(title)}")[:40]
        if not job_id:
            job_id = "uploaded"
        job_id = f"{job_id}_{len(list(JOB_DIR.glob(job_id + '*'))):03d}"
        upload_dir = UPLOAD_DIR / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        audio_path = upload_dir / filename
        with audio_path.open("wb") as handle:
            shutil.copyfileobj(field.file, handle)

        # Start analysis in background thread, return immediately
        _set_job_status(job_id, "queued")
        threading.Thread(
            target=_run_analysis_async,
            args=(audio_path, title, job_id, original_filename),
            daemon=True,
        ).start()
        self.send_json({
            "job_id": job_id,
            "status": "queued",
            "message": "Analysis started. Poll /api/jobs/<job_id>/status for progress.",
        })

    def handle_save_job(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 4:
            self.send_json({"error": "bad_save_path"}, status=400)
            return
        job_id = slugify(parts[2])
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            self.send_json({"error": "missing_body"}, status=400)
            return
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        music_segments = payload.get("music_segments") if isinstance(payload, dict) else []
        timeline = payload.get("timeline") if isinstance(payload, dict) else []
        notes = payload.get("notes") if isinstance(payload, dict) else []
        if not isinstance(music_segments, list) or not isinstance(timeline, list):
            self.send_json({"error": "bad_payload"}, status=400)
            return
        curated = _get_analyzer().save_curated_result(job_id, music_segments, timeline, notes=notes)
        self.send_json(curated)

    def handle_save_custom_action(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            self.send_json({"error": "missing_body"}, status=400)
            return
        if content_length > 1024 * 1024:
            self.send_json({"error": "body_too_large"}, status=413)
            return
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        action = _get_analyzer().save_custom_action(payload)
        self.send_json({"action": action, "status": "saved"})

    def handle_export_video(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            self.send_json({"error": "missing_body"}, status=400)
            return
        if content_length > 10 * 1024 * 1024:
            self.send_json({"error": "body_too_large"}, status=413)
            return
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        result = payload.get("result") if isinstance(payload, dict) else payload
        if not isinstance(result, dict):
            self.send_json({"error": "bad_result"}, status=400)
            return

        audio_path = resolve_result_audio_path(result)
        song = result.get("song") or {}
        song_id = slugify(song.get("song_id") or result.get("job_id") or "yetiger")
        job_id = export_job_id_from_result(result)
        out_dir = JOB_DIR / job_id / "exports"
        out_path = out_dir / f"{song_id}.teaching.mp4"
        _get_video_exporter()(result, audio_path, out_path)
        self.send_file(out_path, download_name=f"{song_id}.teaching.mp4")


def main() -> int:
    # Redirect stderr early
    sys.stderr = sys.stdout
    print("=== YesTiger server starting ===", flush=True)
    print(f"Python: {sys.version}", flush=True)
    parser = argparse.ArgumentParser(description="Run the YesTiger local web app.")
    parser.add_argument("--host", default=os.environ.get("YESTIGER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=default_port())
    args = parser.parse_args()
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOB_DIR.parent / "server-startup.log"
    server = ThreadingHTTPServer((args.host, args.port), YesTigerHandler)
    message = f"YesTiger web app listening on http://{args.host}:{args.port}"
    log_path.write_text(message + "\n", encoding="utf-8")
    print(message, flush=True)
    # Pre-warm heavy imports in background (avoids cold-start timeout on first request)
    threading.Thread(target=_prewarm_analyzer, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping YesTiger web app.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
