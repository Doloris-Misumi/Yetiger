"""Canonical two-level music-structure label schema."""

FINE_LABELS = [
    "bridge",
    "chorus",
    "instrumental_break",
    "intro",
    "outro",
    "post_chorus",
    "pre_chorus",
    "pre_chorus_build",
    "solo",
    "verse",
]

COARSE_LABELS = [
    "bridge",
    "chorus",
    "instrumental",
    "intro",
    "outro",
    "pre_chorus",
    "verse",
]

FINE_TO_COARSE = {
    "bridge": "bridge",
    "chorus": "chorus",
    "instrumental_break": "instrumental",
    "intro": "intro",
    "outro": "outro",
    "post_chorus": "chorus",
    "pre_chorus": "pre_chorus",
    "pre_chorus_build": "pre_chorus",
    "solo": "instrumental",
    "verse": "verse",
}

ANNOTATION_ONLY_LABELS = {"end"}

