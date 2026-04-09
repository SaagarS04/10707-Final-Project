"""
Vocabulary constants shared across model, dataset, and simulator.

All categorical variable lists and result-set definitions live here so there
is a single source of truth for vocab sizes and token indices.
"""

PITCH_TYPES = sorted([
    "CH", "CS", "CU", "EP", "FA", "FC", "FF", "FS",
    "KC", "KN", "SI", "SL", "ST", "SV",
])

PITCH_RESULTS = sorted([
    "ball", "blocked_ball", "called_strike", "foul", "foul_bunt",
    "foul_tip", "hit_by_pitch", "hit_into_play", "missed_bunt",
    "swinging_strike", "swinging_strike_blocked",
    "automatic_ball", "automatic_strike", "pitchout",
    "bunt_foul_tip", "foul_pitchout", "intent_ball",
])

AT_BAT_EVENTS = sorted([
    "catcher_interf", "double", "double_play", "field_error",
    "field_out", "fielders_choice", "fielders_choice_out",
    "force_out", "grounded_into_double_play", "hit_by_pitch",
    "home_run", "intent_walk", "sac_bunt", "sac_bunt_double_play",
    "sac_fly", "sac_fly_double_play", "single", "strikeout",
    "strikeout_double_play", "triple", "triple_play", "truncated_pa",
    "walk",
])

ZONES = list(range(1, 15))  # 1–14

CONTINUOUS_PITCH_COLS = [
    "release_speed", "plate_x", "plate_z", "pfx_x", "pfx_z", "release_spin_rate",
]

# Result sets used by the game engine.
STRIKE_RESULTS = frozenset({
    "called_strike", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "missed_bunt", "automatic_strike",
})
FOUL_RESULTS = frozenset({
    "foul", "foul_bunt", "bunt_foul_tip", "foul_pitchout",
})
BALL_RESULTS = frozenset({
    "ball", "blocked_ball", "automatic_ball", "pitchout", "intent_ball",
})
TERMINAL_RESULTS = frozenset({
    "hit_into_play",
    "hit_by_pitch",
})


def build_vocab_maps() -> tuple[dict, dict, dict, dict]:
    """Build string → index mappings for all categorical variables.

    Returns:
        (pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx)
    """
    pt_to_idx   = {pt: i for i, pt in enumerate(PITCH_TYPES)}
    pr_to_idx   = {pr: i for i, pr in enumerate(PITCH_RESULTS)}
    ev_to_idx   = {ev: i for i, ev in enumerate(AT_BAT_EVENTS)}
    zone_to_idx = {z: i for i, z in enumerate(ZONES)}
    return pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx
