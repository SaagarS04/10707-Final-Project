"""
TransFusion-backed simulator compatible with MHChain.

Wraps the TransFusion model (new_transfusion.py) to expose the three-method
interface that MHChain (mcmc/chain.py) requires:

    simulate_game()           → pregame mode, fresh context
    simulate_from_prefix()    → live-prefix mode, observed prefix encoded
    _simulate_from_state()    → called by simulate_suffix() in MCMC proposals;
                                reads ctx_vecs_at_boundaries[-1] as start context

The key architectural difference from GameSimulator is that TransFusion
updates its context vector per-pitch via _incremental_encode_step rather
than using a constant pregame vector.  We cache the context vector at each
half-inning boundary in GameState.ctx_vecs_at_boundaries so MCMC suffix
proposals can resimulate from any split point without re-encoding the prefix.

Wiring:
    sim = TransFusionSimulator(model, encoders, pitch_scaler, sample, game_pk)
    chain = MHChain(sim, RE24Energy(re24, lam=1.0), SuffixResimulation(sim))
    result = chain.run(n_steps=500, burn_in=100)
    # or for live-prefix:
    result = chain.run_from_prefix(observed_half_innings, n_steps=500)

All mcmc/ components (chain, energy, proposal, acceptance, diagnostics) are
reused unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from sim.types import GameState, HalfInning, AtBatResult, PitchEvent

# ── Imports from teammate's modules ─────────────────────────────────────────
# We import specific names rather than star-importing to make dependencies clear.
from new_transfusion import (
    GameState          as TFGameState,
    EVENT_TABLE        as _TF_EVENT_TABLE,
    IN_PLAY_OUTCOMES   as _IN_PLAY_OUTCOMES,
    _incremental_encode_step,
    _make_context_batch,
    _make_empty_context_batch,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MAX_INNINGS   = 9
_MAX_PA_PITCHES = 30  # safety cap on pitches per plate appearance

# In-play event prior — loaded from empirical cache at simulator init time.
_IN_PLAY_EVENTS: list[str] = [
    "single", "double", "triple", "home_run",
    "field_out", "force_out", "double_play",
    "grounded_into_double_play", "field_error", "sac_fly",
]
# Fallback prior used only if the cache file is missing.
_IN_PLAY_PROBS: np.ndarray = np.array(
    [0.23, 0.06, 0.008, 0.04, 0.45, 0.08, 0.04, 0.03, 0.01, 0.005],
    dtype=np.float64,
)
_IN_PLAY_PROBS = _IN_PLAY_PROBS / _IN_PLAY_PROBS.sum()


def _load_in_play_probs(cache_dir: str) -> None:
    """Load empirical in-play event distribution; updates module globals.

    Priority:
      1. {cache_dir}/in_play_probs.json  — computed from the training split
      2. data/fallback_in_play_probs.json — MLB Statcast 2023 committed to repo
      3. Hard-coded round-number prior    — last resort
    """
    global _IN_PLAY_PROBS
    import json, warnings
    from pathlib import Path

    candidates = [
        Path(cache_dir) / "in_play_probs.json",
        Path(__file__).parent.parent / "data" / "fallback_in_play_probs.json",
    ]

    for path in candidates:
        if path.exists():
            with open(path) as f:
                probs_dict = json.load(f)
            probs = np.array(
                [probs_dict.get(e, 0.0) for e in _IN_PLAY_EVENTS], dtype=np.float64
            )
            s = probs.sum()
            if s > 0:
                probs /= s
            _IN_PLAY_PROBS = probs
            return

    warnings.warn("[in-play] No in_play_probs.json found; using hard-coded fallback prior.")


# ---------------------------------------------------------------------------
# Helper: temperature-scaled multinomial sampling
# ---------------------------------------------------------------------------

def _sample_with_temp(logits: torch.Tensor, temperature: float) -> int:
    """Sample one index from logits with temperature scaling."""
    scaled = logits / max(temperature, 1e-6)
    return int(torch.multinomial(torch.softmax(scaled, dim=-1), 1).item())


def _sample_in_play_event() -> str:
    """Sample a PA-ending event for an in-play outcome using fixed priors."""
    return str(np.random.choice(_IN_PLAY_EVENTS, p=_IN_PLAY_PROBS))


def _get(arr: np.ndarray, idx: int, default: float = 0.0) -> float:
    return float(arr[idx]) if idx < len(arr) else default


# ---------------------------------------------------------------------------
# Helper: build TFGameState from our boundary GameState
# ---------------------------------------------------------------------------

def _build_tf_gs(state: GameState) -> TFGameState:
    """Map our half-inning boundary GameState → TFGameState for the game loop."""
    tf = TFGameState()
    tf.inning     = state.inning
    tf.is_top     = state.is_top
    tf.outs       = state.outs    # 0 at any half-inning boundary
    tf.home_score = state.home_score
    tf.away_score = state.away_score
    tf.on_1b      = state.bases[0]
    tf.on_2b      = state.bases[1]
    tf.on_3b      = state.bases[2]
    tf.balls      = state.balls   # 0 at boundary
    tf.strikes    = state.strikes  # 0 at boundary
    tf.batting_idx = 0
    return tf


# ---------------------------------------------------------------------------
# TransFusionSimulator
# ---------------------------------------------------------------------------

class TransFusionSimulator:
    """Wraps TransFusion to satisfy the GameSimulator interface for MHChain.

    Args:
        model:               TransFusion instance in eval mode.
        encoders:            Encoders instance (from new_dataset_builder).
        pitch_scaler:        StatScaler fit on PITCH_CONTINUOUS_COLS + GAME_STATE_COLS.
        sample:              Dataset sample dict for this game (used for initial
                             context encoding — includes pitcher_ctx, batting_order,
                             game_ctx, etc.).
        game_pk:             MLB game identifier, passed through to GameState.
        device:              Torch device string.
        context_end_idx:     Pitch index marking the end of the observed prefix.
                             0 means pregame (no observed pitches); N means the
                             first N pitches of the game are the observed prefix.
                             This determines what _initial_ctx encodes.
    """

    def __init__(
        self,
        model,
        encoders,
        pitch_scaler,
        sample: dict,
        game_pk: int,
        device: str = "cpu",
        context_end_idx: int = 0,
    ) -> None:
        self.model          = model
        self.encoders       = encoders
        self.pitch_scaler   = pitch_scaler
        self.sample         = sample
        self.game_pk        = game_pk
        self.device         = torch.device(device)
        self.context_end_idx = context_end_idx

        # Reverse vocab maps
        self.idx_to_pt = {v: k for k, v in encoders.pitch_type.items()}
        self.idx_to_oc = {v: k for k, v in encoders.outcome.items()}

        # Ensure model is in inference mode (no dropout, no batch-norm updates).
        self.model.eval()

        # Fixed zero tensors for batter ctx (league-average from scaler means)
        batter_feat_dim = model.cfg.batter_feat_dim
        self._b_ctx_zero  = torch.zeros(1, batter_feat_dim, device=self.device)
        self._bid_zero    = torch.zeros(1, dtype=torch.long, device=self.device)

        # Pre-allocated index tensors reused across pitches to avoid per-pitch allocation.
        self._pt_tok = torch.zeros(1, dtype=torch.long, device=self.device)
        self._oc_tok = torch.zeros(1, dtype=torch.long, device=self.device)

        # Encode and cache the initial context vector (pregame or live-prefix boundary)
        self._initial_ctx = self._encode_initial_context()

    # ── Public API (required by MHChain / simulate_suffix) ──────────────────

    def simulate_game(
        self, temperature: float = 1.0, verbose: bool = False
    ) -> GameState:
        """Pregame simulation: start from the encoded pregame context."""
        init_state = GameState(
            game_pk=self.game_pk,
            inning=1, is_top=True, outs=0, balls=0, strikes=0,
            home_score=0, away_score=0,
            bases=[False, False, False],
            observed_prefix_length=0,
        )
        return self._simulate_from_state_with_ctx(
            init_state, self._initial_ctx, temperature, verbose
        )

    def simulate_from_prefix(
        self,
        observed_half_innings: list[HalfInning],
        temperature: float = 1.0,
        verbose: bool = False,
    ) -> GameState:
        """Live-prefix simulation: observed prefix already encoded in _initial_ctx.

        The observed half-innings are frozen.  MCMC proposals draw split points
        from indices >= len(observed_half_innings).

        The context vectors for the observed prefix are all set to _initial_ctx
        (the encoding of the full observed prefix).  Simulated half-inning
        boundaries get their own per-step contexts appended during simulation.
        """
        home_score = sum(hi.runs for hi in observed_half_innings if not hi.is_top)
        away_score = sum(hi.runs for hi in observed_half_innings if hi.is_top)
        if observed_half_innings:
            last = observed_half_innings[-1]
            next_inning = last.inning if last.is_top else last.inning + 1
            next_is_top = not last.is_top
        else:
            next_inning, next_is_top = 1, True

        # Seed ctx_vecs_at_boundaries for the observed prefix with _initial_ctx
        # (the best single-vector approximation of the context at each boundary).
        obs_ctxs = [self._initial_ctx] * len(observed_half_innings)

        state = GameState(
            game_pk=self.game_pk,
            inning=next_inning, is_top=next_is_top,
            outs=0, balls=0, strikes=0,
            home_score=home_score, away_score=away_score,
            bases=[False, False, False],
            observed_prefix_length=len(observed_half_innings),
            completed_half_innings=list(observed_half_innings),
            ctx_vecs_at_boundaries=obs_ctxs,
        )
        return self._simulate_from_state_with_ctx(
            state, self._initial_ctx, temperature, verbose
        )

    def _simulate_from_state(
        self, state: GameState, temperature: float = 1.0, verbose: bool = False
    ) -> GameState:
        """Called by simulate_suffix() in MCMC proposals.

        Reads ctx_vecs_at_boundaries[-1] as the starting context vector.
        If the list is empty (split_k=0, pregame), falls back to _initial_ctx.
        """
        ctx = (
            state.ctx_vecs_at_boundaries[-1]
            if state.ctx_vecs_at_boundaries
            else self._initial_ctx
        )
        return self._simulate_from_state_with_ctx(state, ctx, temperature, verbose)

    # ── Context encoding ────────────────────────────────────────────────────

    def _encode_initial_context(self) -> torch.Tensor:
        """Encode the game-specific initial context at construction time.

        For pregame (context_end_idx=0): encodes a single dummy step.
        For live-prefix (context_end_idx=N): encodes the first N pitches and
        extracts the context at the last observed pitch position.
        """
        with torch.no_grad():
            if self.context_end_idx == 0:
                batch  = _make_empty_context_batch(self.sample, self.device)
                memory = self.model.encode_context(batch)   # [1, 1, d_model]
                return memory[0, 0, :].detach()
            else:
                batch  = _make_context_batch(
                    self.sample, self.context_end_idx, self.device
                )
                memory = self.model.encode_context(batch)   # [1, T, d_model]
                return memory[0, self.context_end_idx - 1, :].detach()

    # ── Core simulation engine ───────────────────────────────────────────────

    def _simulate_from_state_with_ctx(
        self,
        state: GameState,
        ctx_vec: torch.Tensor,
        temperature: float,
        verbose: bool,
    ) -> GameState:
        """Pitch-by-pitch game loop from a boundary GameState and context vector.

        Builds up completed_half_innings and ctx_vecs_at_boundaries in parallel.
        Returns a fully populated GameState when the game ends.
        """
        tf_gs      = _build_tf_gs(state)
        completed  = list(state.completed_half_innings)
        ctx_vecs   = list(state.ctx_vecs_at_boundaries)
        current_ctx = ctx_vec.unsqueeze(0).to(self.device)  # [1, d_model]

        game_over = False
        while not game_over:
            inning_num   = tf_gs.inning
            is_top_hi    = tf_gs.is_top
            home_before  = tf_gs.home_score
            away_before  = tf_gs.away_score

            at_bats_this_hi: list[AtBatResult] = []
            local_outs = 0
            walk_off   = False

            if verbose:
                side = "Top" if is_top_hi else "Bot"
                print(f"  [{side} {inning_num}] {away_before}–{home_before}")

            while local_outs < 3 and not walk_off:
                ab, outs_added, current_ctx = self._simulate_pa(
                    tf_gs, current_ctx, temperature, verbose
                )
                at_bats_this_hi.append(ab)
                local_outs = min(local_outs + outs_added, 3)

                # Walk-off: home takes lead in bottom of 9th+
                if (
                    not is_top_hi
                    and inning_num >= _MAX_INNINGS
                    and tf_gs.home_score > tf_gs.away_score
                ):
                    walk_off = True

            # Compute runs scored this half-inning from score delta.
            runs = (
                (tf_gs.away_score - away_before) if is_top_hi
                else (tf_gs.home_score - home_before)
            )
            hi = HalfInning(
                inning=inning_num, is_top=is_top_hi,
                at_bats=at_bats_this_hi, runs=runs,
            )
            completed.append(hi)
            ctx_vecs.append(current_ctx[0].detach())

            if walk_off or tf_gs.inning > _MAX_INNINGS:
                game_over = True

        return GameState(
            game_pk=state.game_pk,
            inning=tf_gs.inning, is_top=tf_gs.is_top,
            outs=0, home_score=tf_gs.home_score, away_score=tf_gs.away_score,
            bases=[False, False, False],
            observed_prefix_length=state.observed_prefix_length,
            completed_half_innings=completed,
            ctx_vecs_at_boundaries=ctx_vecs,
        )

    # ── Plate appearance simulation ──────────────────────────────────────────

    def _simulate_pa(
        self,
        tf_gs: TFGameState,
        current_ctx: torch.Tensor,
        temperature: float,
        verbose: bool,
    ) -> tuple[AtBatResult, int, torch.Tensor]:
        """Simulate one plate appearance.

        Returns (AtBatResult, outs_added, updated_ctx_vec [1, d_model]).

        Pitch loop:
          1. Sample pitch type and outcome from model heads (temperature-scaled).
          2. Sample continuous pitch features via DDIM (10 steps).
          3. Apply pitch outcome to TFGameState count.
          4. If in-play outcome: sample PA-ending event from fixed prior.
          5. Update context vector via _incremental_encode_step.
          6. Repeat until terminal event (walk, K, in-play) or 30-pitch cap.

        After the loop, apply the terminal event to TFGameState (updates outs,
        runners, score).  _end_half_inning() is called internally by apply_event
        when outs reach 3, advancing tf_gs to the next half-inning automatically.
        """
        ab_pitches:     list[PitchEvent] = []
        bases_before    = [tf_gs.on_1b, tf_gs.on_2b, tf_gs.on_3b]
        outs_before     = tf_gs.outs
        terminal_event: str | None = None
        last_count      = "0-0"

        for _ in range(_MAX_PA_PITCHES):
            count_before = f"{tf_gs.balls}-{tf_gs.strikes}"

            # ── Step 1: sample pitch type and outcome ─────────────────────
            with torch.no_grad():
                pt_logits, oc_logits = self.model.heads(current_ctx)

            pt_idx = _sample_with_temp(pt_logits[0], temperature)
            oc_idx = _sample_with_temp(oc_logits[0], temperature)
            pt_str = self.idx_to_pt.get(pt_idx, "FF")
            oc_str = self.idx_to_oc.get(oc_idx, "ball")

            self._pt_tok.fill_(pt_idx)
            self._oc_tok.fill_(oc_idx)
            pt_tok = self._pt_tok
            oc_tok = self._oc_tok

            # ── Step 2: sample continuous pitch features (DDIM) ───────────
            with torch.no_grad():
                pitch_feats = self.model.sample_pitch_features(
                    current_ctx, pt_tok, ddim_steps=10
                )  # [1, n_cont]

            # ── Step 3: apply pitch outcome to count ──────────────────────
            event_from_count = tf_gs.apply_pitch_outcome(oc_str)
            count_after      = f"{tf_gs.balls}-{tf_gs.strikes}"
            last_count       = count_after

            if oc_str in _IN_PLAY_OUTCOMES:
                terminal_event = _sample_in_play_event()
            elif event_from_count is not None:
                terminal_event = event_from_count

            # ── Step 4: build game-state feature vector ───────────────────
            gs_feats = torch.tensor(
                tf_gs.to_feature_vec(self.pitch_scaler),
                device=self.device, dtype=torch.float32,
            ).unsqueeze(0)  # [1, F_gs]

            # ── Step 5: record pitch ──────────────────────────────────────
            cont = pitch_feats[0].detach().cpu().numpy()
            ab_pitches.append(PitchEvent(
                pitch_num        = len(ab_pitches) + 1,
                pitch_type       = pt_str,
                zone             = 0,
                result           = oc_str,
                release_speed    = _get(cont, 0),
                plate_x          = _get(cont, 6),
                plate_z          = _get(cont, 7),
                pfx_x            = _get(cont, 4),
                pfx_z            = _get(cont, 5),
                release_spin_rate= _get(cont, 2),
                count_before     = count_before,
                count_after      = count_after,
            ))

            # ── Step 6: update context vector ─────────────────────────────
            with torch.no_grad():
                current_ctx = _incremental_encode_step(
                    model      = self.model,
                    prev_ctx   = current_ctx,
                    pitch_feats= pitch_feats,
                    gs_feats   = gs_feats,
                    b_ctx      = self._b_ctx_zero,
                    batter_id  = self._bid_zero,
                    pt_token   = pt_tok,
                    oc_token   = oc_tok,
                )  # [1, d_model]

            if terminal_event is not None:
                break

        if terminal_event is None:
            terminal_event = "field_out"  # safety cap

        # Apply terminal event: updates outs, runners, score; may call _end_half_inning.
        tf_gs.apply_event(terminal_event)
        outs_added = _TF_EVENT_TABLE.get(terminal_event, (1, 0, "out"))[0]

        if verbose:
            print(f"      {terminal_event} ({last_count})")

        ab = AtBatResult(
            pitches      = ab_pitches,
            event        = terminal_event,
            final_count  = last_count,
            bases_before = bases_before,
            outs_before  = outs_before,
        )
        return ab, outs_added, current_ctx
