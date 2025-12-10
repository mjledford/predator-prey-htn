import random
import gtpyhop

from constants import DO_NOTHING, PREY
from plan_utils import build_planner_state, joint_plan_to_actions

class CommStats:
    """Track communication and replanning events for evaluation."""
    def __init__(self):
        self.messages = 0   # abstract messages (obs + actions)
        self.replans = 0    # number of planner calls


class HTNCommModule:
    """
    Communication controller between centralized HTN planner and agents.

    Modes:
    - 'full': replan every step.
    - 'periodic': replan every k steps.
    - 'event': replan only when trigger condition met.
    - 'none': no replanning; agents reuse fixed random plan forever.
    """
    
    def __init__(self, mode="full", k_sync=5, debug=False):
        assert mode in ("full", "periodic", "event", "none"), f"Unknown mode: {mode}"
        self.mode = mode
        self.k_sync = k_sync
        self.debug = debug

        self.stats = CommStats()
        self._cached_actions = None  # stores most recent joint plan
        self._frozen_plan = None     # used for 'none' mode baseline

    def _should_replan(self, t: int, event_triggered: bool) -> bool:
        if self.mode == "full":
            return True
        if self.mode == "periodic":
            return (t % self.k_sync == 0)
        if self.mode == "event":
            return event_triggered or (t % 10 == 0)  # fallback every 10 steps
        return False  # for "none"

    def _compute_event_trigger(self, observations) -> bool:
        triggered = any(PREY in obs for obs in observations.values())
        if self.debug:
            print(f"[COMM] Event trigger = {triggered}")
        return triggered

    def _build_htn_state(self, env, observations, agent_memory, keep_prev_action):
        agent_ids = list(env.agents)
        s = build_planner_state(env, observations)
        s.prev_actions = {aid: agent_memory[aid]["prev_action"] for aid in agent_ids}
        s.keep_prev_action = keep_prev_action
        s.rngs = {aid: agent_memory[aid]["rng"] for aid in agent_ids}
        s.agent_ids = agent_ids
        return s

    def decide_actions(self, t, env, observations, agent_memory, keep_prev_action):
        """
        Decide actions based on current communication mode.

        Returns:
            actions: dict[agent_id] -> action int
        """
        agent_ids = list(env.agents)
        M = len(agent_ids)
        event_triggered = self._compute_event_trigger(observations)

        if self.mode == "none":
            if self._frozen_plan is None:
                s = self._build_htn_state(env, observations, agent_memory, keep_prev_action)
                plan = gtpyhop.find_plan(s, [("choose_joint_action", tuple(env.agents))])
                self._frozen_plan = joint_plan_to_actions(plan, list(env.agents))
                self.stats.messages += 2 * len(env.agents)
                self.stats.replans += 1
            return self._frozen_plan

        if not self._should_replan(t, event_triggered) and self._cached_actions is not None:
            if self.debug:
                print(f"[COMM] t={t}: reuse cached plan (mode={self.mode}, event={event_triggered})")
            return self._cached_actions

        # --- Replanning path ---
        s = self._build_htn_state(env, observations, agent_memory, keep_prev_action)
        plan = gtpyhop.find_plan(s, [("choose_joint_action", tuple(agent_ids))])
        actions = joint_plan_to_actions(plan, agent_ids)

        self.stats.replans += 1
        self.stats.messages += 2 * M
        self._cached_actions = actions

        if self.debug:
            print(f"[COMM] t={t}: REPLAN (mode={self.mode}, event={event_triggered}) -> actions = {actions}")

        return actions
