from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parent.parent


def processed_call_audio(action_id: str) -> Path:
    return ROOT / "call_audio" / action_id / "processed" / f"{action_id}_ota_throat_grit_m2.wav"


CALL_AUDIO_FILES: Dict[str, Path] = {
    "standard_mix": ROOT / "call_audio" / "standard_mix" / "processed" / "standard_mix.wav",
    "standard_mix_long": processed_call_audio("standard_mix_long"),
    "standard_mix_first_half": processed_call_audio("standard_mix_first_half"),
    "standard_mix_second_half": processed_call_audio("standard_mix_second_half"),
    "mix_leadin_aaa_ikuzo": processed_call_audio("mix_leadin_aaa_ikuzo"),
    "japanese_mix": processed_call_audio("japanese_mix"),
    "japanese_mix_long": processed_call_audio("japanese_mix_long"),
    "japanese_mix_second_half": processed_call_audio("japanese_mix_second_half"),
    "ainu_mix": processed_call_audio("ainu_mix"),
    "ainu_second_half_mix": processed_call_audio("ainu_second_half_mix"),
    "ainu_kahen_mix": processed_call_audio("ainu_kahen_mix"),
    "myhontousuke": processed_call_audio("myhontousuke"),
    "myohon_activation": processed_call_audio("myohon_activation"),
    "kaho_sanren_mix": processed_call_audio("kaho_sanren_mix"),
    "ietora_konzetsu_mix": processed_call_audio("ietora_konzetsu_mix"),
    "ietora": processed_call_audio("ietora"),
    "ietora_long": processed_call_audio("ietora_long"),
    "tiger_fire_activation": processed_call_audio("tiger_fire_activation"),
    "bismarck_mix": processed_call_audio("bismarck_mix"),
    "bismarck_mix_first_half": processed_call_audio("bismarck_mix_first_half"),
    "sekai_konton_mix": processed_call_audio("sekai_konton_mix"),
    "sekai_konton_mix_first_half": processed_call_audio("sekai_konton_mix_first_half"),
    "bandor_mix": processed_call_audio("bandor_mix"),
    "popipa_mix": processed_call_audio("popipa_mix"),
    "lin_xiu_mix": processed_call_audio("lin_xiu_mix"),
    "bariyado_mix": processed_call_audio("bariyado_mix"),
    "pan_mix": processed_call_audio("pan_mix"),
    "aiai_mix": processed_call_audio("aiai_mix"),
    "imi_fumei_ai_mix": processed_call_audio("imi_fumei_ai_mix"),
}


def call_audio_path(action_id: str) -> Optional[Path]:
    return CALL_AUDIO_FILES.get(str(action_id or ""))
