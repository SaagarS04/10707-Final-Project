"""
MH chain runner.

Responsibility: wire together energy, proposal, and acceptance into a
runnable Markov chain.  No mathematical logic lives here — it belongs in
energy.py, proposal.py, and acceptance.py respectively.

Two entry points:

    MHChain.run()
        Run the chain from a fresh simulated game (pregame mode).

    MHChain.run_from_prefix()
        Run the chain conditioned on an observed game prefix (live-game mode,
        Option A: half-inning boundary conditioning).
        The observed prefix is frozen; proposals only modify the suffix.
"""

from __future__ import annotations

import random

from sim.simulator import GameSimulator
from sim.types import HalfInning
from mcmc.types import Trajectory, ChainSample, ChainResult
from mcmc.energy import Energy
from mcmc.proposal import Proposal
from mcmc.acceptance import log_acceptance_ratio, accept
import mcmc.diagnostics as diagnostics_mod


class MHChain:
    """Metropolis-Hastings chain over game trajectories.

    Args:
        simulator:   GameSimulator used to initialize the chain and within
                     the proposal kernel (via SuffixResimulation).
        energy:      Energy function E(τ).  Use RE24Energy or any Energy impl.
        proposal:    Proposal kernel.  Use SuffixResimulation or any Proposal impl.
        temperature: Simulator sampling temperature.
    """

    def __init__(
        self,
        simulator: GameSimulator,
        energy: Energy,
        proposal: Proposal,
        temperature: float = 1.0,
    ):
        self.simulator   = simulator
        self.energy      = energy
        self.proposal    = proposal
        self.temperature = temperature

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        n_steps: int = 500,
        burn_in: int = 100,
        run_diagnostics: bool = True,
    ) -> ChainResult:
        """Run the chain from a fresh simulated game (pregame mode).

        Args:
            n_steps:         Number of post-burn-in steps to collect.
            burn_in:         Number of initial steps to discard.
            run_diagnostics: If True, compute ESS, autocorrelation, etc.

        Returns:
            ChainResult with win_probability, acceptance_rate, samples, diagnostics.
        """
        initial = self._init_trajectory()
        return self._run_chain(initial, n_steps, burn_in, run_diagnostics)

    def run_from_prefix(
        self,
        observed_half_innings: list[HalfInning],
        n_steps: int = 500,
        burn_in: int = 100,
        run_diagnostics: bool = True,
    ) -> ChainResult:
        """Run the chain conditioned on an observed game prefix (live-game mode).

        The observed prefix is frozen permanently.  Proposals only modify
        half-innings at indices >= len(observed_half_innings).

        Args:
            observed_half_innings: Completed half-innings already played.
            n_steps:               Post-burn-in steps.
            burn_in:               Burn-in steps.
            run_diagnostics:       Compute diagnostics.

        Returns:
            ChainResult.
        """
        initial_game = self.simulator.simulate_from_prefix(
            observed_half_innings, temperature=self.temperature, verbose=False
        )
        initial = Trajectory(state=initial_game)
        initial.log_energy = self.energy(initial)
        return self._run_chain(initial, n_steps, burn_in, run_diagnostics)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal chain loop
    # ──────────────────────────────────────────────────────────────────────────

    def _init_trajectory(self) -> Trajectory:
        state = self.simulator.simulate_game(temperature=self.temperature, verbose=False)
        traj  = Trajectory(state=state)
        traj.log_energy = self.energy(traj)
        return traj

    def _run_chain(
        self,
        current: Trajectory,
        n_steps: int,
        burn_in: int,
        run_diagnostics: bool,
    ) -> ChainResult:
        samples: list[ChainSample] = []
        n_accepted = 0

        for step in range(burn_in + n_steps):
            proposed, split_k, n_valid_curr, n_valid_prop = self.proposal.propose(
                current, temperature=self.temperature
            )

            # Compute suffix energies (prefix energies cancel — see acceptance.py).
            log_alpha = log_acceptance_ratio(
                current=current,
                proposed=proposed,
                split_k=split_k,
                suffix_energy_fn=self.energy.suffix,
                n_valid_current=n_valid_curr,
                n_valid_proposed=n_valid_prop,
            )

            accepted = accept(log_alpha, rng_uniform=random.random())
            if accepted:
                proposed.log_energy = self.energy(proposed)
                current = proposed

            if step >= burn_in:
                if accepted:
                    n_accepted += 1
                samples.append(ChainSample(
                    home_wins=current.home_wins,
                    n_half_innings=current.n_half_innings,
                    home_score=current.state.home_score,
                    away_score=current.state.away_score,
                    log_energy=current.log_energy,
                    accepted=accepted,
                ))

        win_probability  = float(sum(s.home_wins for s in samples)) / max(len(samples), 1)
        acceptance_rate  = n_accepted / max(n_steps, 1)

        diag = {}
        if run_diagnostics and samples:
            diag = diagnostics_mod.compute(samples)

        return ChainResult(
            win_probability=win_probability,
            acceptance_rate=acceptance_rate,
            n_samples=len(samples),
            samples=samples,
            diagnostics=diag,
        )
