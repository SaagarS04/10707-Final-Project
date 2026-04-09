"""
Shared types for the game simulator and MCMC sampler.

This module is the sole shared dependency between sim/ and mcmc/.
Neither package imports the other — they communicate only through these types.

Types exported:
    PitchEvent   — one pitch outcome (discrete + continuous attributes)
    GameState    — complete serializable game state at any point in a game
    HalfInning   — a completed half-inning (list of at-bat results + score change)
    AtBatResult  — a completed at-bat (list of pitches + terminal event)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PitchEvent:
    """One pitch within an at-bat.

    Attributes:
        pitch_num:      1-based index within the at-bat.
        pitch_type:     MLB pitch type code (e.g. 'FF', 'SL').
        zone:           Statcast zone number (1–14).
        result:         Pitch outcome string (e.g. 'called_strike', 'ball').
        release_speed:  Pitch velocity in mph (denormalized).
        plate_x:        Horizontal plate position in feet.
        plate_z:        Vertical plate position in feet.
        pfx_x:          Horizontal movement in inches.
        pfx_z:          Vertical movement in inches.
        release_spin_rate: Spin rate in rpm.
        count_before:   Count string before this pitch (e.g. '1-2').
        count_after:    Count string after this pitch.
        from_prefix:    True if this pitch came from an observed prefix (not simulated).
    """
    pitch_num: int
    pitch_type: str
    zone: int
    result: str
    release_speed: float
    plate_x: float
    plate_z: float
    pfx_x: float
    pfx_z: float
    release_spin_rate: float
    count_before: str
    count_after: str
    from_prefix: bool = False


@dataclass
class AtBatResult:
    """A completed plate appearance.

    Attributes:
        pitches:       Ordered list of PitchEvent objects.
        event:         Terminal at-bat event (e.g. 'single', 'strikeout', 'walk').
        final_count:   Count at the end of the at-bat (e.g. '3-2').
        bases_before:  Base occupancy at start of at-bat [1B, 2B, 3B].
        outs_before:   Out count at start of at-bat.
    """
    pitches: list[PitchEvent]
    event: str
    final_count: str
    bases_before: list[bool] = field(default_factory=lambda: [False, False, False])
    outs_before: int = 0


@dataclass
class HalfInning:
    """A completed half-inning.

    Attributes:
        inning:    Inning number (1-indexed).
        is_top:    True if the visiting team batted (top half).
        at_bats:   Ordered list of AtBatResult objects.
        runs:      Runs scored in this half-inning.
    """
    inning: int
    is_top: bool
    at_bats: list[AtBatResult]
    runs: int


@dataclass
class GameState:
    """Complete serializable game state at any point in a game.

    This is the shared currency between sim/ and mcmc/. MCMC suffix proposals
    reconstruct game state at a half-inning boundary from this object.

    Attributes:
        game_pk:                  MLB unique game identifier.
        inning:                   Current inning (1-indexed).
        is_top:                   True if top of inning (away batting).
        outs:                     Current out count (0, 1, or 2).
        home_score:               Home team score.
        away_score:               Away team score.
        bases:                    [on_1b, on_2b, on_3b] boolean runner occupancy.
        balls:                    Current ball count (0–3).
        strikes:                  Current strike count (0–2).
        current_ab_history:       Pitches seen so far in the current at-bat.
        observed_prefix_length:   Number of completed half-innings that are
                                  observed (fixed) rather than simulated.
                                  0 for pregame mode; k for live-game mode
                                  conditioned on k completed half-innings.
        completed_half_innings:   Half-innings already played (observed or simulated).
    """
    game_pk: int
    inning: int
    is_top: bool
    outs: int
    home_score: int
    away_score: int
    bases: list[bool]               # [on_1b, on_2b, on_3b]
    balls: int = 0
    strikes: int = 0
    current_ab_history: list[PitchEvent] = field(default_factory=list)
    observed_prefix_length: int = 0
    completed_half_innings: list[HalfInning] = field(default_factory=list)
    # Context vectors at each half-inning boundary, parallel to completed_half_innings.
    # ctx_vecs_at_boundaries[i] is the model context vector after completing half-inning i.
    # Used by TransFusionSimulator so MCMC proposals can resimulate from any split point
    # without re-encoding the full prefix.  Empty list for GameSimulator (constant context).
    ctx_vecs_at_boundaries: list = field(default_factory=list)

    @property
    def n_half_innings(self) -> int:
        """Total number of completed half-innings (observed + simulated)."""
        return len(self.completed_half_innings)

    @property
    def score_diff_batting(self) -> float:
        """Score differential from the batting team's perspective."""
        return (self.away_score - self.home_score) if self.is_top \
               else (self.home_score - self.away_score)

    def at_inning_boundary(self) -> "GameState":
        """Return a copy reset to the start of the next at-bat at the current
        half-inning boundary (outs=0, bases clear, count 0-0, no AB history).
        Used by MCMC to splice a new suffix at a half-inning split point.
        """
        import copy as _copy
        s = _copy.copy(self)
        s.outs = 0
        s.balls = 0
        s.strikes = 0
        s.bases = [False, False, False]
        s.current_ab_history = []
        return s
