import random
import gtpyhop

from constants import DO_NOTHING, PREY
from plan_utils import build_planner_state, joint_plan_to_actions

class CommStats:
    """Track simple communication and replanning stats"""
    def __init__(self):
        self.messages = 0 #abstract messages between planner and agents
        self.replans = 0 # number of planner calls

class HTNCommModule:
    """
    Communication module between agents and centralized HTN planner.

    For now: full communication every step.
    Later: you can add modes for periodic/event-triggered comm.
    """
    
    def __init__(self, mode="full", k_sync=5, debug : bool = False):
        assert mode in ("full", "periodic", "event"), f"Unknown mode {mode}"
        self.mode = mode
        self.k_sync = k_sync
        self.debug = debug
        
        self._last_actions = None #cache last actions from last plan
        self.stats = CommStats()
    
    def _should_replan(self, t: int, event_triggered: bool) -> bool:
        if self.mode == "full":
            return True
        if self.mode == "periodic":
            return (t % self.k_sync == 0)
        if self.mode == "event":
            return event_triggered
        return True
    
    def _compute_event_trigger(self, observation) -> bool:
        """
        Simple event trigger:
         - True if ANY agent currently sees PREY in its local obs
         
        TODO: Refine this later
        """
        for obs in observation.values():
            if PREY in obs:
                return True
        return False
        
    
    def decide_actions(self, t, env, observations, agent_memory, keep_prev_action: bool):
        """
        Compute joint actions for this step.

        Full communication model:
          - All agents send their obs to planner
          - Planner replans using joint HTN
          - Planner sends actions to all agents

        Returns:
            actions: dict[agent_id] -> action int
        """
        agent_ids = list(env.agents)
        M = len(agent_ids)
        
        # Event trigger: any agent currently sees prey
        event_triggered = self._compute_event_trigger(observations)
        
        # Decide whether to replan this step
        if not self._should_replan(t, event_triggered) and self._last_actions is not None:
            if self.debug:
                print(f"[COMM] t={t}: reuse previous actions (mode={self.mode}, event={event_triggered})")
            return self._last_actions
        
        # --- Replanning path ---
        if self.debug:
            print(f"[COMM] t={t}: REPLAN (mode={self.mode}, event={event_triggered})")

        # Conceptual message counting: M obs up + M actions down
        self.stats.messages += 2 * M
        self.stats.replans += 1

        # Build HTN state from observations
        s = build_planner_state(env, observations)

        # Inject per-agent memory & flags into state for planner use
        s.prev_actions = {aid: agent_memory[aid]["prev_action"] for aid in agent_ids}
        s.keep_prev_action = keep_prev_action
        s.rngs = {aid: agent_memory[aid]["rng"] for aid in agent_ids}
        s.agent_ids = agent_ids

        # Single joint HTN call
        plan = gtpyhop.find_plan(s, [("choose_joint_action", tuple(agent_ids))])
        actions = joint_plan_to_actions(plan, agent_ids)

        if self.debug:
            print(f"[COMM] t={t}: joint actions = {actions}")

        self._last_actions = actions
        return actions
    
    
        