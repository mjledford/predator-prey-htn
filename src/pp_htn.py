import gtpyhop
import random



from constants import (
    DO_NOTHING, UP, DOWN, LEFT, RIGHT,
    EMPTY, WALL, PRED, PREY,
    DIRS, ORDERED_DIRS
)

from pp_behavior import (
    action_from_obs,
    legal_moves_from_obs,
    find_global_leader,
    choose_leader_action,
    choose_helper_action,
    choose_patrol_action,
)

DEBUG = False

# ----------------------------------------------------------------------
# Primitive action
# ----------------------------------------------------------------------
def do(state, agent_id, action_id):
    """
    Record the chosen discrete action for agent_id
    The simiulator (gym env) will eventually execute it outsideGTPyhop
    """
    if not hasattr(state, "last_action"):
        state.last_action = {}
    state.last_action[agent_id] = int(action_id)
    return state # returning the (modified) state signals success




# ----------------------------------------------------------------------
# Legacy single-agent HTN methods (baseline, not used in joint planner)
# ----------------------------------------------------------------------
def _default_fallback_action():
    return DO_NOTHING

def _default_fallback_plan(agent_id):
    return [("do", agent_id, DO_NOTHING)]  

   
# METHODS for 'choose_action'
def m_chase_if_visible(state, agent_id):
    """
    Legacy single-agent method:
    If prey visible in local obs, chase it; else fail so next method can try.
    """
    if DEBUG:
        print("[DEBUG] [legacy] Chase (prey visible)")

    obs = state.obs[agent_id]
    action = action_from_obs(obs, state.obs_dim)
    if action == DO_NOTHING:
        return False

    if DEBUG:
        print(f"[DEBUG] [legacy] Agent {agent_id} chase action {action}")
    return [("do", agent_id, action)]

def m_patrol_if_not_visible(state, agent_id):
    """
    Legacy single-agent patrol using the shared choose_patrol_action helper.
    Uses unified state conventions: state.prev_actions, state.keep_prev_action, state.rngs.
    """
    if DEBUG:
        print(f"[DEBUG] [legacy] Patrol (prey not visible)")

    action = choose_patrol_action(state, agent_id)
    return [("do", agent_id, action)]

# ----------------------------------------------------------------------
# Joint HTN method (this is what we actually use)
# ----------------------------------------------------------------------
def m_choose_joint_action(state, agent_ids):
    """
    Joint HTN method:
      - If any predator sees prey:
          * pick a leader (closest to prey)
          * leader chases
          * helpers coordinate based on leader
      - Else:
          * everyone patrols using shared patrol logic.
    """
    obs_dict = state.obs
    obs_dim = state.obs_dim

    subtasks = []

    # 1) Leader selection
    leader, dx, dy = find_global_leader(obs_dict, agent_ids, obs_dim)

    if leader is not None:
        # Prey visible: leader + helpers
        leader_action = choose_leader_action(state, leader)
        subtasks.append(("do", leader, leader_action))

        for aid in agent_ids:
            if aid == leader:
                continue
            helper_action = choose_helper_action(state, aid, leader_action)
            subtasks.append(("do", aid, helper_action))

    else:
        # No prey visible: each agent patrols
        for aid in agent_ids:
            a = choose_patrol_action(state, aid)
            subtasks.append(("do", aid, a))

    return subtasks

# ----------------------------------------------------------------------
# Domain registration
# ----------------------------------------------------------------------
domain_name = "pp_htn"

# Importing this module will create and register the "pp_htn" domain.
gtpyhop.Domain(domain_name)
gtpyhop.declare_actions(do)


# Progress Report 1: Individual actions
# Legacy single-agent API (keep as baseline, currently NOT used by run_demo):
#gtpyhop.declare_task_methods("choose_action", m_chase_if_visible, m_patrol_if_not_visible)

# Progress Report 2: Joint actions
# Joint planner API (this is what run_demo should call):
gtpyhop.declare_task_methods( "choose_joint_action", m_choose_joint_action)
