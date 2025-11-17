import gtpyhop

from constants import DO_NOTHING


def plan_to_actions(plan):
    """Plan should be in the list[tuple] for: [('do', 0, 4)] --> agent 0 do action 4"""
    if not plan:
        return DO_NOTHING
    op, aid, act = plan[0]
    return int(act)
    # for op, aid, act in plan:
    #     return act

def joint_plan_to_actions(plan, agent_ids):
    """
    Convert a joint GTPyhop plan into a per-agent action dict.

    Plan is expected to be a list of ('do', agent_id, action) tuples.
    We take the FIRST primitive action for each agent.
    """
    actions = {aid: DO_NOTHING for aid in agent_ids}

    if not plan:
        return actions

    for op, aid, act in plan:
        if op != "do":
            continue
        if aid in actions and actions[aid] == DO_NOTHING:
            actions[aid] = int(act)
        # once everybody has a first action, stop
        if all(actions[aid] != DO_NOTHING for aid in agent_ids):
            break

    return actions


def build_planner_state(env, observations):
    """Build a GTPyhop state from the current environment observations."""
    s = gtpyhop.State("tick")
    s.obs = {agent_id: observations[agent_id] for agent_id in env.agents }
    s.obs_dim = env.unwrapped.model.obs_dim
    
    return s
