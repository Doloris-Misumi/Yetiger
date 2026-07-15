"""Process recorded call/MIX audio into the YesTiger ota_throat_grit_m2 style.

The outputs are 48 kHz stereo WAV files intended for browser preview and
backend MP4 muxing.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT / ".venv" / "Scripts" / "ffmpeg.exe"

SOURCE_MAP: Dict[str, str] = {
    # Previously processed
    "standard_mix": "standard_mix.m4a",
    "japanese_mix": "Japanese_mix.m4a",
    "ainu_mix": "ainu_mix.m4a",
    "pan_mix": "pan_mix.m4a",
    "aiai_mix": "aiai_mix.m4a",
    "imi_fumei_ai_mix": "意味不明.m4a",
    "kaho_sanren_mix": "可变三连.m4a",
    "ietora_konzetsu_mix": "家虎根绝.m4a",
    "tiger_fire_activation": "虎虎发动.m4a",
    # New recordings (2026-07-12)
    "standard_mix_long": "standard_mix_long.m4a",
    "standard_mix_first_half": "standard_mix_first_half.m4a",
    "standard_mix_second_half": "standard_mix_second_half.m4a",
    "mix_leadin_aaa_ikuzo": "mix_leadin_aaa_ikuzo.m4a",
    "japanese_mix_long": "japanese_mix_long.m4a",
    "japanese_mix_second_half": "japanese_mix_second_half.m4a",
    "ainu_second_half_mix": "ainu_second_half_mix.m4a",
    "ainu_kahen_mix": "ainu_kahen_mix.m4a",
    "myhontousuke": "myhontousuke.m4a",
    "myohon_activation": "myohon_activation.m4a",
    "ietora": "ietora.m4a",
    "ietora_long": "ietora_long.m4a",
    "bismarck_mix": "bismarck_mix.m4a",
    "bismarck_mix_first_half": "bismarck_mix_first_half.m4a",
    "sekai_konton_mix": "sekai_konton_mix.m4a",
    "sekai_konton_mix_first_half": "sekai_konton_mix_first_half.m4a",
    "bandor_mix": "bandor_mix.m4a",
    "popipa_mix": "popipa_mix.m4a",
    "lin_xiu_mix": "lin_xiu_mix.m4a",
    "bariyado_mix": "bariyado_mix.m4a",
}

# Approximate the existing standard_mix/processed/ota_throat_grit_m2 character:
# slightly raised pitch, reduced chest mud, boosted nasal/presence bands,
# firm compression, a very small amount of controlled grit, then loudness/peak
# normalization close to the reference file's -13 dB mean and -2.5 dB peak.
OTA_THROAT_GRIT_M2_FILTER = ",".join(
    [
        "pan=stereo|c0=c0|c1=c0",
        "volume=-1dB",
        "rubberband=pitch=1.055",
        "highpass=f=130",
        "lowpass=f=11500",
        "equalizer=f=250:t=q:w=1.0:g=-4",
        "equalizer=f=1500:t=q:w=1.0:g=2.5",
        "equalizer=f=3100:t=q:w=1.1:g=4.5",
        "equalizer=f=5600:t=q:w=1.0:g=2",
        "acompressor=threshold=0.08:ratio=5:attack=3:release=65:makeup=4",
        "acrusher=level_in=1.15:level_out=0.80:bits=13:mix=0.12:mode=log",
        "alimiter=limit=0.78:level=false",
        "loudnorm=I=-10.5:TP=-2.5:LRA=7",
        "volume=6dB",
        "alimiter=limit=0.75:level=false",
    ]
)

# loudnorm can severely under-normalize very short burst calls such as ietora.
# Keep the same pitch/EQ/compression/grit character, then use limiter-based
# peak control instead of measured loudness normalization for those clips.
OTA_THROAT_GRIT_M2_SHORT_FILTER = ",".join(
    [
        "pan=stereo|c0=c0|c1=c0",
        "volume=-1dB",
        "rubberband=pitch=1.055",
        "highpass=f=130",
        "lowpass=f=11500",
        "equalizer=f=250:t=q:w=1.0:g=-4",
        "equalizer=f=1500:t=q:w=1.0:g=2.5",
        "equalizer=f=3100:t=q:w=1.1:g=4.5",
        "equalizer=f=5600:t=q:w=1.0:g=2",
        "acompressor=threshold=0.08:ratio=5:attack=3:release=65:makeup=4",
        "acrusher=level_in=1.15:level_out=0.80:bits=13:mix=0.12:mode=log",
        "volume=6dB",
        "alimiter=limit=0.75:level=false",
    ]
)

FILTER_OVERRIDES: Dict[str, str] = {
    "ietora": OTA_THROAT_GRIT_M2_SHORT_FILTER,
}


def resolve_ffmpeg() -> str:
    if FFMPEG.exists():
        return str(FFMPEG)
    env_ffmpeg = Path(sys.executable).with_name("ffmpeg.exe")
    if env_ffmpeg.exists():
        return str(env_ffmpeg)
    resolved = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if resolved:
        return resolved
    raise FileNotFoundError("ffmpeg executable not found")


def process_action(action_id: str, source_name: str, root: Path = ROOT) -> Path:
    source = root / "call_audio" / source_name
    if not source.exists():
        raise FileNotFoundError(source)

    output_dir = root / "call_audio" / action_id / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{action_id}_ota_throat_grit_m2.wav"
    filter_graph = FILTER_OVERRIDES.get(action_id, OTA_THROAT_GRIT_M2_FILTER)

    cmd = [
        resolve_ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-af",
        filter_graph,
        "-ar",
        "48000",
        "-ac",
        "2",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root containing call_audio/")
    parser.add_argument("--action", action="append", choices=sorted(SOURCE_MAP), help="Process only this action id. Repeatable.")
    args = parser.parse_args()

    root = args.root.resolve()
    action_ids = args.action or list(SOURCE_MAP)
    for action_id in action_ids:
        source_name = SOURCE_MAP[action_id]
        output = process_action(action_id, source_name, root=root)
        print(f"{action_id}: {source_name} -> {output.relative_to(root)}")


if __name__ == "__main__":
    main()
