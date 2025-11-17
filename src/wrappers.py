import posggym
import time

ACTION_NAME = {0:"STAY", 1:"UP", 2:"DOWN", 3:"LEFT", 4:"RIGHT"}




def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class ActionLoggingWrapper(posggym.Wrapper):
    def __init__(self, env, debug=True, log_every=1):
        super().__init__(env)
        self.debug = debug
        self.t = 0
        self.log_every = log_every
        # --- stats ---
        self._t0 = None
        self._prev_prey_caught = None         # tuple[int,...]
        self.capture_events = []               # list of dicts
        self.episode_rewards = {}              # per-agent cumulative
        self.first_capture_step = None

    def reset(self, *args, **kwargs):
        obs, infos = self.env.reset(*args, **kwargs)
        self.t = 0
        self._t0 = time.time()
        # prime previous flags from current state
        self._prev_prey_caught = tuple(self.unwrapped.state[2])
        self.capture_events.clear()
        self.episode_rewards = {i: 0.0 for i in self.unwrapped.model.possible_agents}
        self.first_capture_step = None
        if self.debug:
            preds = tuple(self.unwrapped.state[0])
            preys = tuple(self.unwrapped.state[1])
            print(f"[reset] preds={preds} preys={preys} prey_caught={self._prev_prey_caught}, predator_strength={self.unwrapped.model.prey_strength}    ")
        return obs, infos

    def _log_capture_events(self, rewards, next_state):
        """Detect new captures, log step, coord, involved predators, and timing."""
        curr_flags = tuple(next_state[2])
        prev_flags = self._prev_prey_caught
        if curr_flags == prev_flags:
            return  # no new captures this step

        preds = tuple(next_state[0])
        preys = tuple(next_state[1])
        prey_strength = self.unwrapped.model.prey_strength

        for i, (before, after) in enumerate(zip(prev_flags, curr_flags)):
            if before == 0 and after == 1:
                prey_coord = preys[i]
                # who was adjacent this step?
                involved_pred_idx = [pi for pi, pc in enumerate(preds) if manhattan(pc, prey_coord) <= 1]
                # wall-clock and step timing
                wall_sec = None if self._t0 is None else (time.time() - self._t0)
                event = {
                    "step": self.t,
                    "wall_time_sec": wall_sec,
                    "prey_index": i,
                    "prey_coord": prey_coord,
                    "involved_predators": involved_pred_idx,
                    "num_adjacent": len(involved_pred_idx),
                    "prey_strength": prey_strength,
                    "rewards_this_step": rewards,  # per-agent dict
                }
                self.capture_events.append(event)
                if self.first_capture_step is None:
                    self.first_capture_step = self.t
                if self.debug:
                    print(f"  >>> CAPTURE @ step={self.t} prey={i} coord={prey_coord} "
                          f"involved={involved_pred_idx} (need {prey_strength})")
        self._prev_prey_caught = curr_flags

    def step(self, actions):
        # (optional) pre-step log
        if self.debug and (self.t % self.log_every == 0):
            a_str = {aid: ACTION_NAME.get(a, a) for aid, a in actions.items()}
            preds_before = tuple(self.unwrapped.state[0])
            preys_before = tuple(self.unwrapped.state[1])
            print(f"[t={self.t:03d}] actions={a_str}")
            print(f"         before preds={preds_before} preys={preys_before}")

        obs, rewards, term, trunc, done, infos = self.env.step(actions)

        # accumulate rewards
        for aid, r in rewards.items():
            self.episode_rewards[aid] = self.episode_rewards.get(aid, 0.0) + float(r)

        # post-step state & logs
        preds_after = tuple(self.unwrapped.state[0])
        preys_after = tuple(self.unwrapped.state[1])
        if self.debug and (self.t % self.log_every == 0):
            print(f"         rewards={rewards} term={term} trunc={trunc} all_done={done}")
            print(f"         after  preds={preds_after}  preys={preys_after}")

        # detect and record capture events
        self._log_capture_events(rewards, self.unwrapped.state)

        # episode summary when done
        if done:
            total_wall = None if self._t0 is None else (time.time() - self._t0)
            if self.debug:
                print("\n=== EPISODE SUMMARY ===")
                print(f"steps_taken={self.t+1} "
                      f"first_capture_step={self.first_capture_step} "
                      f"wall_time_sec≈{total_wall:.3f}" if total_wall is not None else "")
                print(f"total_rewards={self.episode_rewards}")
                print("captures:")
                for ev in self.capture_events:
                    print(f"  - step={ev['step']} t≈{(ev['wall_time_sec'] or 0):.3f}s "
                          f"prey={ev['prey_index']} at {ev['prey_coord']} "
                          f"involved={ev['involved_predators']} "
                          f"adj={ev['num_adjacent']}/{ev['prey_strength']} "
                          f"rewards={ev['rewards_this_step']}")

        self.t += 1
        return obs, rewards, term, trunc, done, infos


