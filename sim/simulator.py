"""
At-bat-local GameSimulator.

Simulates a full MLB game pitch-by-pitch using PitchSequenceTransfusion.
History (pitch embeddings fed back as input) resets to <start> tokens at
every at-bat boundary, matching the at-bat-local training contract of
AtBatSequenceDataset.

Exports:
    GameSimulator          — full game and live-prefix simulation.
    simulate_suffix        — simulate from a given GameState (used by MCMC).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from model.vocab import (
    PITCH_TYPES, ZONES, PITCH_RESULTS, AT_BAT_EVENTS,
    CONTINUOUS_PITCH_COLS,
    STRIKE_RESULTS, FOUL_RESULTS, BALL_RESULTS,
)
from sim.types import GameState, HalfInning, AtBatResult, PitchEvent


class GameSimulator:
    """Full MLB game simulator using the Transfusion pitch model.

    Simulates pitch-by-pitch with proper MLB rules:
      - Count tracking (balls, strikes, fouls)
      - 3-out half-innings, top/bottom alternation
      - 9 innings; extra innings with ghost runner on 2B (post-2020 rule)
      - Walk on 4 balls, strikeout on 3 strikes
      - At-bat event prediction when pitch_result is terminal

    History contract: pitch history resets to <start> at every at-bat boundary.
    This matches AtBatSequenceDataset and must not be changed without also
    updating the canonical training dataset.

    Args:
        model:            Trained PitchSequenceTransfusion instance (eval mode).
        context_features: Normalized pregame context vector (context_dim,).
        pt_to_idx:        pitch_type → index mapping.
        pr_to_idx:        pitch_result → index mapping.
        ev_to_idx:        at_bat_event → index mapping.
        zone_to_idx:      zone → index mapping.
        pitch_mean:       Mean used for continuous feature normalization (n_cont,).
        pitch_std:        Std used for continuous feature normalization (n_cont,).
        game_pk:          MLB game identifier (passed through to GameState).
        device:           Torch device string.
    """

    def __init__(
        self,
        model,
        context_features: np.ndarray,
        pt_to_idx: dict, pr_to_idx: dict, ev_to_idx: dict, zone_to_idx: dict,
        pitch_mean: np.ndarray, pitch_std: np.ndarray,
        game_pk: int = 0,
        device: str = "cpu",
    ):
        self.model = model
        self.context_features = context_features
        self.pt_to_idx   = pt_to_idx
        self.pr_to_idx   = pr_to_idx
        self.ev_to_idx   = ev_to_idx
        self.zone_to_idx = zone_to_idx
        self.idx_to_pt   = {v: k for k, v in pt_to_idx.items()}
        self.idx_to_pr   = {v: k for k, v in pr_to_idx.items()}
        self.idx_to_ev   = {v: k for k, v in ev_to_idx.items()}
        self.idx_to_zone = {v: k for k, v in zone_to_idx.items()}
        self.pitch_mean  = pitch_mean
        self.pitch_std   = pitch_std
        self.game_pk     = game_pk
        self.device      = torch.device(device)
        self.model.eval()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def simulate_game(self, temperature: float = 1.0, verbose: bool = False) -> GameState:
        """Simulate a full game from first pitch.

        Returns:
            GameState with all completed half-innings, final score, and
            observed_prefix_length=0 (pregame mode).
        """
        state = GameState(
            game_pk=self.game_pk,
            inning=1, is_top=True,
            outs=0, balls=0, strikes=0,
            home_score=0, away_score=0,
            bases=[False, False, False],
            observed_prefix_length=0,
        )
        return self._simulate_from_state(state, temperature=temperature, verbose=verbose)

    def simulate_from_prefix(
        self,
        observed_half_innings: list[HalfInning],
        temperature: float = 1.0,
        verbose: bool = False,
    ) -> GameState:
        """Simulate the suffix of a game from a given observed prefix.

        The observed prefix is frozen: MCMC proposals may only modify
        half-innings at indices >= len(observed_half_innings).

        Args:
            observed_half_innings: List of HalfInning objects that have already
                                   been played (Option A: half-inning boundary).
            temperature:           Sampling temperature.
            verbose:               Print play-by-play.

        Returns:
            GameState with observed_prefix_length set to len(observed_half_innings),
            and completed_half_innings = observed_prefix + simulated_suffix.
        """
        # Reconstruct score from prefix.
        home_score = away_score = 0
        for hi in observed_half_innings:
            if hi.is_top:
                away_score += hi.runs
            else:
                home_score += hi.runs

        # Determine the next half-inning to simulate.
        if observed_half_innings:
            last = observed_half_innings[-1]
            if last.is_top:
                next_inning, next_is_top = last.inning, False
            else:
                next_inning, next_is_top = last.inning + 1, True
        else:
            next_inning, next_is_top = 1, True

        state = GameState(
            game_pk=self.game_pk,
            inning=next_inning, is_top=next_is_top,
            outs=0, balls=0, strikes=0,
            home_score=home_score, away_score=away_score,
            bases=[False, False, False],
            observed_prefix_length=len(observed_half_innings),
            completed_half_innings=list(observed_half_innings),
        )
        return self._simulate_from_state(state, temperature=temperature, verbose=verbose)

    # ──────────────────────────────────────────────────────────────────────────
    # Core simulation engine
    # ──────────────────────────────────────────────────────────────────────────

    def _simulate_from_state(
        self, state: GameState, temperature: float, verbose: bool
    ) -> GameState:
        """Drive the game loop from state until a winner is determined."""
        MAX_INNINGS = 9
        inning      = state.inning
        is_top      = state.is_top
        home_score  = state.home_score
        away_score  = state.away_score
        half_innings = list(state.completed_half_innings)

        while True:
            extra = inning > MAX_INNINGS

            if verbose:
                side = "Top" if is_top else "Bottom"
                batting = away_score if is_top else home_score
                fielding = home_score if is_top else away_score
                print(f"\n{'='*50}")
                print(f"  {side} of Inning {inning} | Away: {away_score}  Home: {home_score}")
                print(f"{'='*50}")

            hi, runs = self._simulate_half_inning(
                inning=inning, is_top=is_top,
                home_score=home_score, away_score=away_score,
                extra_runner=extra,
                temperature=temperature, verbose=verbose,
            )
            half_innings.append(hi)

            if is_top:
                away_score += runs
            else:
                home_score += runs

            # Walk-off: home takes lead in bottom of 9th+.
            if not is_top and inning >= MAX_INNINGS and home_score > away_score:
                if verbose:
                    print("  *** WALK-OFF! ***")
                break

            # End-of-inning logic.
            if is_top:
                # Still need bottom half unless home already leads (mercy rule doesn't apply here).
                is_top = False
            else:
                # Completed both halves of the inning.
                if inning >= MAX_INNINGS and home_score != away_score:
                    break  # regulation or extra-innings winner
                inning += 1
                is_top = True

        if verbose:
            print(f"\n{'='*50}")
            print(f"  FINAL: Away {away_score}  Home {home_score}")
            print(f"{'='*50}")

        return GameState(
            game_pk=state.game_pk,
            inning=inning, is_top=is_top,
            outs=0, home_score=home_score, away_score=away_score,
            bases=[False, False, False],
            observed_prefix_length=state.observed_prefix_length,
            completed_half_innings=half_innings,
        )

    def _simulate_half_inning(
        self, inning: int, is_top: bool, home_score: int, away_score: int,
        extra_runner: bool = False, temperature: float = 1.0, verbose: bool = False,
    ) -> tuple[HalfInning, int]:
        """Simulate one half-inning until 3 outs."""
        outs  = 0
        runs  = 0
        bases = [False, False, False]
        at_bats: list[AtBatResult] = []

        if extra_runner:
            bases[1] = True
            if verbose:
                print("  [Extra innings: ghost runner on 2B]")

        while outs < 3:
            bases_before = list(bases)
            outs_before  = outs

            ab = self._simulate_at_bat(
                inning=inning, is_top=is_top, outs=outs,
                home_score=home_score, away_score=away_score,
                bases=bases, temperature=temperature, verbose=verbose,
            )
            ab.bases_before = bases_before
            ab.outs_before  = outs_before
            at_bats.append(ab)

            new_outs, new_runs, bases = _apply_event(ab.event, outs, bases)
            outs  = new_outs
            runs += new_runs

            if verbose:
                base_str = "".join(
                    f"{b}B " for b, occ in zip([1, 2, 3], bases) if occ
                ) or "empty"
                print(f"  → {ab.event} | Outs: {outs} | Runs: {runs} | Bases: {base_str}")

            # Walk-off check (live).
            if not is_top and inning >= 9 and (home_score + runs) > away_score:
                if verbose:
                    print("  *** WALK-OFF! ***")
                break

        return HalfInning(inning=inning, is_top=is_top, at_bats=at_bats, runs=runs), runs

    def _simulate_at_bat(
        self, inning: int, is_top: bool, outs: int,
        home_score: int, away_score: int, bases: list[bool],
        temperature: float = 1.0, verbose: bool = False,
    ) -> AtBatResult:
        """Simulate one at-bat pitch-by-pitch.

        History resets to <start> tokens at the start of every at-bat.
        This is the canonical at-bat-local inference contract.
        """
        balls = strikes = 0
        score_diff = (away_score - home_score) if is_top else (home_score - away_score)

        # Reset history (at-bat-local contract).
        prev_pt   = len(PITCH_TYPES)
        prev_zone = len(ZONES)
        prev_pr   = len(PITCH_RESULTS)
        prev_cont = np.zeros(len(CONTINUOUS_PITCH_COLS), dtype=np.float32)

        history_pt   = []
        history_zone = []
        history_pr   = []
        history_cont = []
        history_gs   = []
        pitch_log: list[PitchEvent] = []
        event = None

        with torch.no_grad():
            for pitch_num in range(30):  # safety cap
                gs = np.array([
                    balls, strikes, outs, inning / 9.0, score_diff / 10.0,
                    float(bases[0]), float(bases[1]), float(bases[2]), float(is_top),
                ], dtype=np.float32)

                history_pt.append(prev_pt)
                history_zone.append(prev_zone)
                history_pr.append(prev_pr)
                history_cont.append(prev_cont.copy())
                history_gs.append(gs)
                S = len(history_pt)

                ctx  = torch.tensor(self.context_features, dtype=torch.float32) \
                             .unsqueeze(0).unsqueeze(0).expand(1, S, -1).to(self.device)
                gs_t = torch.tensor(np.array(history_gs),   dtype=torch.float32).unsqueeze(0).to(self.device)
                pt_t = torch.tensor(history_pt,             dtype=torch.long).unsqueeze(0).to(self.device)
                z_t  = torch.tensor(history_zone,           dtype=torch.long).unsqueeze(0).to(self.device)
                pr_t = torch.tensor(history_pr,             dtype=torch.long).unsqueeze(0).to(self.device)
                c_t  = torch.tensor(np.array(history_cont), dtype=torch.float32).unsqueeze(0).to(self.device)

                out = self.model(ctx, gs_t, pt_t, z_t, pr_t, c_t)

                sampled_pt   = _sample(out["pitch_type_logits"][0, -1],   temperature)
                sampled_zone = _sample(out["zone_logits"][0, -1],          temperature)
                sampled_pr   = _sample(out["pitch_result_logits"][0, -1], temperature)

                # DDPM continuous sampling.
                cond = out["continuous_latent"][:, :S, :].permute(0, 2, 1)
                seq_length = self.model.ddpm.seq_length
                if cond.shape[2] < seq_length:
                    cond = F.pad(cond, (0, seq_length - cond.shape[2]))
                elif cond.shape[2] > seq_length:
                    cond = cond[:, :, :seq_length]
                sampled_seq = self.model.ddpm.sample(batch_size=1, cond=cond)
                pos_idx     = min(S - 1, seq_length - 1)
                sampled_cont_norm = sampled_seq[0, :, pos_idx].cpu().numpy()
                raw_cont = sampled_cont_norm * self.pitch_std + self.pitch_mean

                pt_str     = self.idx_to_pt.get(sampled_pt, "FF")
                zone_val   = self.idx_to_zone.get(sampled_zone, 5)
                result_str = self.idx_to_pr.get(sampled_pr, "ball")

                count_before = f"{balls}-{strikes}"
                if result_str in STRIKE_RESULTS:
                    strikes += 1
                elif result_str in FOUL_RESULTS:
                    if strikes < 2:
                        strikes += 1
                elif result_str in BALL_RESULTS:
                    balls += 1
                count_after = f"{balls}-{strikes}"

                if verbose:
                    print(f"    P{pitch_num+1}: {pt_str} {raw_cont[0]:.1f}mph → {result_str} ({count_after})")

                pitch_log.append(PitchEvent(
                    pitch_num=pitch_num + 1,
                    pitch_type=pt_str, zone=zone_val, result=result_str,
                    release_speed=float(raw_cont[0]),
                    plate_x=float(raw_cont[1]), plate_z=float(raw_cont[2]),
                    pfx_x=float(raw_cont[3]), pfx_z=float(raw_cont[4]),
                    release_spin_rate=float(raw_cont[5]),
                    count_before=count_before, count_after=count_after,
                ))

                if strikes >= 3:
                    event = "strikeout"; break
                if balls >= 4:
                    event = "walk"; break
                if result_str == "hit_by_pitch":
                    event = "hit_by_pitch"; break
                if result_str == "hit_into_play":
                    ev_idx = _sample(out["at_bat_event_logits"][0, -1], temperature)
                    event  = self.idx_to_ev.get(ev_idx, "field_out")
                    break

                prev_pt   = sampled_pt
                prev_zone = sampled_zone
                prev_pr   = sampled_pr
                prev_cont = sampled_cont_norm

        if event is None:
            event = "field_out"

        return AtBatResult(pitches=pitch_log, event=event, final_count=f"{balls}-{strikes}")


# ──────────────────────────────────────────────────────────────────────────────
# Public convenience: simulate suffix from a GameState (used by mcmc/proposal.py)
# ──────────────────────────────────────────────────────────────────────────────

def simulate_suffix(simulator: GameSimulator, state: GameState, temperature: float = 1.0) -> GameState:
    """Simulate from a half-inning boundary GameState to game end.

    This is the primitive used by the MCMC proposal kernel. The caller is
    responsible for ensuring state represents a valid half-inning boundary
    (outs=0, bases clear, count 0-0).

    Args:
        simulator:   Configured GameSimulator instance.
        state:       GameState at the split point (boundary of observed prefix).
        temperature: Sampling temperature.

    Returns:
        A new GameState with completed_half_innings = state.completed_half_innings
        + newly simulated suffix.
    """
    return simulator._simulate_from_state(state, temperature=temperature, verbose=False)


# ──────────────────────────────────────────────────────────────────────────────
# MLB game-logic helpers (pure functions — no model dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _sample(logits: torch.Tensor, temperature: float) -> int:
    """Multinomial sample from temperature-scaled logits."""
    return torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).item()


def _apply_event(event: str, outs: int, bases: list[bool]) -> tuple[int, int, list[bool]]:
    """Apply an at-bat event. Returns (new_outs, runs_scored, new_bases).

    This is a simplified but rule-correct baserunner model. All base
    advancement follows the most common MLB rule interpretation.
    """
    runs  = 0
    bases = list(bases)

    if event in ("strikeout", "strikeout_double_play"):
        outs += 1
        if event == "strikeout_double_play":
            outs += 1

    elif event == "field_out":
        outs += 1
        if bases[2]: runs += 1; bases[2] = False
        if bases[1]: bases[2] = True; bases[1] = False

    elif event == "force_out":
        outs += 1

    elif event == "grounded_into_double_play":
        outs += 2
        if bases[0]: bases[0] = False

    elif event in ("double_play",):
        outs += 2

    elif event == "triple_play":
        outs += 3

    elif event == "single":
        if bases[2]: runs += 1; bases[2] = False
        if bases[1]: bases[2] = True; bases[1] = False
        if bases[0]: bases[1] = True; bases[0] = False
        bases[0] = True

    elif event == "double":
        if bases[2]: runs += 1; bases[2] = False
        if bases[1]: runs += 1; bases[1] = False
        if bases[0]: bases[2] = True; bases[0] = False
        bases[1] = True

    elif event == "triple":
        runs += sum(bases)
        bases = [False, False, False]
        bases[2] = True

    elif event == "home_run":
        runs += sum(bases) + 1
        bases = [False, False, False]

    elif event in ("walk", "hit_by_pitch", "intent_walk", "catcher_interf"):
        if bases[0] and bases[1] and bases[2]:
            runs += 1
        elif bases[0] and bases[1]:
            bases[2] = True
        elif bases[0]:
            bases[1] = True
        bases[0] = True

    elif event == "sac_fly":
        outs += 1
        if bases[2]: runs += 1; bases[2] = False

    elif event == "sac_fly_double_play":
        outs += 2
        if bases[2]: runs += 1; bases[2] = False

    elif event == "sac_bunt":
        outs += 1
        if bases[2]: runs += 1; bases[2] = False
        if bases[1]: bases[2] = True; bases[1] = False
        if bases[0]: bases[1] = True; bases[0] = False

    elif event == "sac_bunt_double_play":
        outs += 2

    elif event == "field_error":
        if bases[2]: runs += 1; bases[2] = False
        if bases[1]: bases[2] = True; bases[1] = False
        if bases[0]: bases[1] = True; bases[0] = False
        bases[0] = True

    elif event in ("fielders_choice", "fielders_choice_out"):
        outs += 1
        bases[0] = True

    # truncated_pa and unknown events: no change.

    outs = min(outs, 3)
    return outs, runs, bases
