
from typing import Dict, Tuple, Any

# cell codes in Predator-Prey
EMPTY, WALL, PREDATOR, PREY = 0, 1, 2, 3

class HelloObserver:
    """
    Minimal observer that 'receives' per-agent observations each step and prints
    a compact summary. Great for smoke tests or wiring a planner later.
    """

    def __init__(self, pretty: bool = False):
        self.pretty = pretty
        self.obs_dim = None
        self.step = 0

    # ---- lifecycle hooks --------------------------------------------------

    def on_reset(self, env, observations: Dict[str, Tuple[int, ...]], infos: Dict[str, Any]):
        """Call after env.reset(...) to initialize internal state."""
        self.step = 0
        # auto-detect obs_dim from env (works with Predator-Prey)
        self.obs_dim = env.unwrapped.model.obs_dim
        size = 2 * self.obs_dim + 1
        print(f"[observer] reset: agents={env.agents}, obs_dim={self.obs_dim} (window {size}x{size})")

        # (optional) show initial obs summary
        self._print_obs_summary(observations)

    def on_step(self, t: int, observations: Dict[str, Tuple[int, ...]], rewards, terminations, truncations, infos):
        """Call once per tick with the latest observations and signals."""
        self.step = t
        self._print_obs_summary(observations)

    def on_episode_end(self, reason: str):
        """Call when episode finishes."""
        print(f"[observer] episode ended at step {self.step} ({reason})")

    # ---- helpers ----------------------------------------------------------

    def _print_obs_summary(self, observations: Dict[str, Tuple[int, ...]]):
        size = 2 * self.obs_dim + 1
        for aid, obs in observations.items():
            # count visible entities
            n_prey = sum(1 for v in obs if v == PREY)
            n_pred = sum(1 for v in obs if v == PREDATOR) - 1  # minus self at center
            # compute relative offsets to any visible prey
            prey_offsets = self._prey_offsets(obs, size)
            print(f"[observer:t={self.step:02d}] agent={aid} sees prey={n_prey} pred_others={n_pred} prey_offsets={prey_offsets}")

            if self.pretty:
                print(self._pretty_obs(obs, size))

    @staticmethod
    def _prey_offsets(obs: Tuple[int, ...], size: int):
        # center index for egocentric window
        center = size // 2
        offs = []
        for k, v in enumerate(obs):
            if v == PREY:
                r, c = divmod(k, size)
                dx, dy = (c - center), (r - center)  # +x = right, -x = left, +y down, -y = up
                offs.append((dx, dy))
        return offs

    @staticmethod
    def _pretty_obs(obs: Tuple[int, ...], size: int) -> str:
        rows = [obs[i:i+size] for i in range(0, size*size, size)]
        return "\n".join(" ".join(map(str, row)) for row in rows)
