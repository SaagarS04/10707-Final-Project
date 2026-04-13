"""
Unit tests for GameState, _phi, and empirical table loading.

Run with:  pytest tests/test_game_state.py -v
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

# Make sure the project root is on sys.path regardless of where pytest is invoked.
sys.path.insert(0, str(Path(__file__).parent.parent))

from new_transfusion import (
    GameState,
    _phi,
    _RE24_TABLE_CACHE,
    _RE24_MAX_CACHE,
    _load_re24_table,
    _load_in_play_probs,
    _IN_PLAY_EVENTS_ARR,
    _IN_PLAY_PROBS_ARR,
    _IN_PLAY_CUMPROBS,
)


# =============================================================================
# GameState — outs and half-inning transitions
# =============================================================================

class TestOutsAndHalfInnings:
    def test_three_outs_flips_to_bottom(self):
        gs = GameState()
        assert gs.is_top and gs.inning == 1
        gs.apply_event("field_out")
        gs.apply_event("field_out")
        gs.apply_event("field_out")
        assert not gs.is_top
        assert gs.inning == 1
        assert gs.outs == 0

    def test_six_outs_advances_inning(self):
        gs = GameState()
        for _ in range(6):
            gs.apply_event("field_out")
        assert gs.is_top and gs.inning == 2

    def test_double_play_adds_two_outs(self):
        gs = GameState()
        gs.on_1b = True
        gs.apply_event("double_play")
        assert gs.outs == 2

    def test_outs_reset_after_half_inning(self):
        gs = GameState()
        gs.apply_event("field_out")
        gs.apply_event("field_out")
        gs.apply_event("field_out")
        assert gs.outs == 0

    def test_bases_cleared_after_half_inning(self):
        gs = GameState()
        gs.on_1b = gs.on_2b = gs.on_3b = True
        gs.apply_event("field_out")
        gs.apply_event("field_out")
        gs.apply_event("field_out")
        assert not gs.on_1b and not gs.on_2b and not gs.on_3b


# =============================================================================
# GameState — scoring
# =============================================================================

class TestScoring:
    def test_home_run_scores_all_runners(self):
        gs = GameState()
        gs.is_top = False  # home team batting
        gs.on_1b = gs.on_2b = gs.on_3b = True
        runs = gs.apply_event("home_run")
        assert runs == 4
        assert gs.home_score == 4
        assert not gs.on_1b and not gs.on_2b and not gs.on_3b

    def test_away_team_scores_to_away_score(self):
        gs = GameState()  # is_top=True → away batting
        gs.on_3b = True
        gs.apply_event("single")
        assert gs.away_score == 1
        assert gs.home_score == 0

    def test_walk_with_bases_loaded_scores_run(self):
        gs = GameState()
        gs.on_1b = gs.on_2b = gs.on_3b = True
        runs = gs.apply_walk()
        assert runs == 1
        assert gs.on_1b and gs.on_2b and gs.on_3b

    def test_walk_no_force_does_not_score(self):
        gs = GameState()
        gs.on_1b = True
        runs = gs.apply_walk()
        assert runs == 0
        assert gs.on_1b and gs.on_2b

    def test_sac_fly_scores_one_adds_out(self):
        gs = GameState()
        gs.on_3b = True
        runs = gs.apply_event("sac_fly")
        assert runs == 1
        assert gs.outs == 1


# =============================================================================
# GameState — runner advancement
# =============================================================================

class TestRunnerAdvancement:
    def test_single_advances_all_runners_one(self):
        gs = GameState()
        gs.on_1b = True
        gs.apply_event("single")
        assert gs.on_1b  # batter on 1st
        assert gs.on_2b  # runner advanced to 2nd

    def test_double_scores_runner_from_first(self):
        gs = GameState()
        gs.on_1b = True
        runs = gs.apply_event("double")
        assert gs.on_2b   # batter on 2nd
        assert gs.on_3b   # runner from 1st goes to 3rd
        assert runs == 0  # runner from 1st doesn't score on a double

    def test_triple_scores_all_runners(self):
        gs = GameState()
        gs.on_1b = gs.on_2b = gs.on_3b = True
        runs = gs.apply_event("triple")
        assert runs == 3
        assert gs.on_3b   # batter on 3rd
        assert not gs.on_1b and not gs.on_2b


# =============================================================================
# GameState — is_game_over
# =============================================================================

class TestIsGameOver:
    def _make_post_bottom_9th(self, home, away):
        """Return a GameState that looks like the bottom of the 9th just ended."""
        gs = GameState()
        gs.inning    = 10   # _end_half_inning advanced it from 9 → 10
        gs.is_top    = True
        gs.home_score = home
        gs.away_score = away
        return gs

    def test_not_over_before_9th(self):
        gs = GameState()
        gs.inning = 8
        gs.is_top = True
        gs.home_score = 5
        gs.away_score = 3
        assert not gs.is_game_over()

    def test_not_over_mid_inning(self):
        gs = GameState()
        gs.inning = 9
        gs.is_top = False   # bottom of 9th still in progress
        gs.home_score = 5
        gs.away_score = 3
        assert not gs.is_game_over()

    def test_over_after_bottom_9th_home_leads(self):
        gs = self._make_post_bottom_9th(home=3, away=2)
        assert gs.is_game_over()

    def test_over_after_bottom_9th_away_leads(self):
        gs = self._make_post_bottom_9th(home=2, away=5)
        assert gs.is_game_over()

    def test_not_over_after_bottom_9th_tied(self):
        gs = self._make_post_bottom_9th(home=3, away=3)
        assert not gs.is_game_over()

    def test_not_over_exactly_at_min_innings(self):
        # inning == min_innings (not > min_innings) → not over yet
        gs = GameState()
        gs.inning    = 9
        gs.is_top    = True   # just flipped (bottom of 8th done, now top of 9th)
        gs.home_score = 5
        gs.away_score = 3
        assert not gs.is_game_over()


# =============================================================================
# GameState — walk-off
# =============================================================================

class TestWalkoff:
    def test_walkoff_set_in_bottom_9th(self):
        gs = GameState()
        gs.inning  = 9
        gs.is_top  = False
        gs.away_score = 3
        gs.home_score = 3
        gs.on_3b = True
        gs.apply_event("single")
        assert gs._check_walkoff()

    def test_no_walkoff_in_early_innings(self):
        gs = GameState()
        gs.inning  = 5
        gs.is_top  = False
        gs.away_score = 0
        gs.on_3b = True
        gs.apply_event("single")
        assert not gs._check_walkoff()

    def test_no_walkoff_if_home_does_not_lead(self):
        gs = GameState()
        gs.inning  = 9
        gs.is_top  = False
        gs.away_score = 5
        gs.home_score = 3
        gs.on_3b = True
        gs.apply_event("single")  # now 4-5, still behind
        assert not gs._check_walkoff()


# =============================================================================
# GameState — extra innings
# =============================================================================

class TestExtraInnings:
    def test_automatic_runner_on_2nd_in_extras(self):
        gs = GameState()
        # simulate 3 outs in bottom of 9th  → inning becomes 10
        gs.inning  = 9
        gs.is_top  = False
        gs.apply_event("field_out")
        gs.apply_event("field_out")
        gs.apply_event("field_out")
        assert gs.inning == 10
        assert gs.is_top
        assert gs.on_2b  # automatic runner

    def test_no_automatic_runner_in_regular_innings(self):
        gs = GameState()
        # 3 outs in top of 1st
        gs.apply_event("field_out")
        gs.apply_event("field_out")
        gs.apply_event("field_out")
        assert not gs.on_2b


# =============================================================================
# _phi — RE24 energy function
# =============================================================================

class TestPhi:
    def test_phi_nonpositive(self):
        """φ(s) ≤ 0 for every state."""
        for outs in range(3):
            for b1 in (False, True):
                for b2 in (False, True):
                    for b3 in (False, True):
                        assert _phi(outs, b1, b2, b3) <= 0.0

    def test_phi_max_state_is_zero(self):
        """The best state (bases loaded, 0 outs) should give φ ≈ 0."""
        phi_max = _phi(0, True, True, True)
        assert abs(phi_max) < 1e-9

    def test_phi_worst_state_most_negative(self):
        """Empty bases, 2 outs should be the most negative φ."""
        phi_worst = _phi(2, False, False, False)
        phi_best  = _phi(0, True, True, True)
        assert phi_worst < phi_best

    def test_phi_monotone_with_runners(self):
        """More runners → higher RE24 → less negative φ."""
        phi_empty = _phi(0, False, False, False)
        phi_1b    = _phi(0, True,  False, False)
        assert phi_1b > phi_empty

    def test_phi_unknown_state_returns_floor(self):
        """Unknown state key should not crash; returns log(1e-6) - log(max)."""
        val = _phi(99, False, False, False)
        expected = math.log(1e-6) - math.log(_RE24_MAX_CACHE)
        assert abs(val - expected) < 1e-9


# =============================================================================
# _load_re24_table — loader from parquet
# =============================================================================

class TestLoadRE24Table:
    def test_fallback_when_file_missing(self):
        """Missing parquet should leave the cache with its default fallback values."""
        import new_transfusion as nt
        original = dict(nt._RE24_TABLE_CACHE)
        original_max = nt._RE24_MAX_CACHE
        with tempfile.TemporaryDirectory() as tmp:
            _load_re24_table(tmp)  # file doesn't exist
        assert nt._RE24_TABLE_CACHE == original
        assert nt._RE24_MAX_CACHE == original_max

    def test_loads_from_parquet(self):
        import pandas as pd
        import new_transfusion as nt

        rows = []
        for outs in range(3):
            for b1 in (False, True):
                for b2 in (False, True):
                    for b3 in (False, True):
                        rows.append({"outs": outs, "on_1b": b1, "on_2b": b2,
                                     "on_3b": b3, "re24_value": 1.23})
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            df.to_parquet(f"{tmp}/re24_table.parquet", index=False)
            _load_re24_table(tmp)
        assert nt._RE24_TABLE_CACHE[(0, False, False, False)] == pytest.approx(1.23)
        assert nt._RE24_MAX_CACHE == pytest.approx(1.23)


# =============================================================================
# _load_in_play_probs — loader from JSON
# =============================================================================

class TestLoadInPlayProbs:
    def test_falls_through_to_repo_fallback_when_cache_missing(self, tmp_path):
        """No cache file → loader should load the repo fallback, not the round-number prior."""
        import new_transfusion as nt
        from pathlib import Path
        repo_fallback = Path(__file__).parent.parent / "data" / "fallback_in_play_probs.json"
        if not repo_fallback.exists():
            pytest.skip("data/fallback_in_play_probs.json not present")
        _load_in_play_probs(str(tmp_path))
        # Probs should now match the repo fallback, not the hard-coded round numbers
        with open(repo_fallback) as f:
            expected = json.load(f)
        idx = list(_IN_PLAY_EVENTS_ARR).index("field_out")
        assert nt._IN_PLAY_PROBS_ARR[idx] == pytest.approx(expected["field_out"], rel=1e-5)

    def test_loads_and_normalizes(self, tmp_path):
        import numpy as np
        import new_transfusion as nt

        # Write unnormalized counts
        probs = {e: 10.0 for e in _IN_PLAY_EVENTS_ARR}
        (tmp_path / "in_play_probs.json").write_text(json.dumps(probs))
        _load_in_play_probs(str(tmp_path))
        assert abs(nt._IN_PLAY_PROBS_ARR.sum() - 1.0) < 1e-9

    def test_cumprobs_monotone_and_ends_at_one(self, tmp_path):
        import numpy as np
        import new_transfusion as nt

        probs = {str(e): float(i + 1) for i, e in enumerate(_IN_PLAY_EVENTS_ARR)}
        (tmp_path / "in_play_probs.json").write_text(json.dumps(probs))
        _load_in_play_probs(str(tmp_path))
        assert nt._IN_PLAY_CUMPROBS[-1] == pytest.approx(1.0)
        assert all(
            nt._IN_PLAY_CUMPROBS[i] <= nt._IN_PLAY_CUMPROBS[i + 1]
            for i in range(len(nt._IN_PLAY_CUMPROBS) - 1)
        )

    def test_missing_event_keys_default_to_zero(self, tmp_path):
        import new_transfusion as nt

        # Only provide "single"; everything else should be 0 before normalization
        (tmp_path / "in_play_probs.json").write_text(json.dumps({"single": 1.0}))
        _load_in_play_probs(str(tmp_path))
        # After normalization "single" should be 1.0
        idx = list(_IN_PLAY_EVENTS_ARR).index("single")
        assert nt._IN_PLAY_PROBS_ARR[idx] == pytest.approx(1.0)

    def test_repo_fallback_loaded_when_cache_missing(self, tmp_path):
        """When no cache file exists, loader should pick up the repo fallback."""
        import new_transfusion as nt
        from pathlib import Path

        repo_fallback = Path(__file__).parent.parent / "data" / "fallback_in_play_probs.json"
        if not repo_fallback.exists():
            pytest.skip("data/fallback_in_play_probs.json not present")

        # tmp_path has no in_play_probs.json → should load from data/fallback
        _load_in_play_probs(str(tmp_path))
        # Verify values match the repo file
        with open(repo_fallback) as f:
            expected = json.load(f)
        idx = list(_IN_PLAY_EVENTS_ARR).index("single")
        assert nt._IN_PLAY_PROBS_ARR[idx] == pytest.approx(expected["single"], rel=1e-5)

    def test_repo_fallback_values_are_real_data(self):
        """Sanity-check the committed fallback: sums to 1, no event is zero."""
        from pathlib import Path
        repo_fallback = Path(__file__).parent.parent / "data" / "fallback_in_play_probs.json"
        if not repo_fallback.exists():
            pytest.skip("data/fallback_in_play_probs.json not present")
        with open(repo_fallback) as f:
            probs = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        assert abs(sum(probs.values()) - 1.0) < 1e-4
        for event in _IN_PLAY_EVENTS_ARR:
            assert probs.get(event, 0.0) > 0.0, f"{event} has zero probability in fallback"
