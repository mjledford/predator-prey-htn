import random

from constants import (
    DO_NOTHING, UP, DOWN, LEFT, RIGHT,
    EMPTY, WALL, PRED, PREY,
    DIRS, ORDERED_DIRS
)

def action_from_obs(obs, obs_dim):
    """
    Greedy chase if any PREY cells visible in local obs window.
    """
    size = 2 * obs_dim + 1
    center = obs_dim

    prey_idxs = [k for k, v in enumerate(obs) if v == PREY]
    if not prey_idxs:
        return DO_NOTHING

    best_md = 10**9
    best_dx = 0
    best_dy = 0
    for k in prey_idxs:
        r, c = divmod(k, size)
        dx = c - center         # +x = right
        dy = r - center         # +y = down
        md = abs(dx) + abs(dy)
        if md < best_md:
            best_md = md
            best_dx, best_dy = dx, dy

    dx, dy = best_dx, best_dy

    # move greedily to reduce distance; break ties horizontally
    if abs(dx) >= abs(dy):
        if dx > 0:  return RIGHT
        if dx < 0:  return LEFT
        if dy > 0:  return DOWN
        if dy < 0:  return UP
        return DO_NOTHING
    else:
        if dy > 0:  return DOWN
        if dy < 0:  return UP
        if dx > 0:  return RIGHT
        if dx < 0:  return LEFT
        return DO_NOTHING


def legal_moves_from_obs(obs, obs_dim):
    """
    Given local obs, return list of legal move action_ids (no WALL).
    """
    size = 2 * obs_dim + 1
    center = obs_dim
    legal_moves = []
    for a in ORDERED_DIRS:
        dx, dy = DIRS[a]
        nx, ny = center + dx, center + dy
        if not (0 <= nx < size and 0 <= ny < size):
            continue
        idx = ny * size + nx
        if obs[idx] != WALL:
            legal_moves.append(a)
    return legal_moves


def find_global_leader(obs_dict, agent_ids, obs_dim):
    """
    Scan all agents' local obs and find:
      - agent_id of closest predator to any visible prey
      - (dx, dy) from that predator to the chosen prey
    Returns:
      (best_agent, best_dx, best_dy) or (None, 0, 0) if no prey visible.
    """
    best_agent = None
    best_md = 10**9
    best_dx = 0
    best_dy = 0

    size = 2 * obs_dim + 1
    center = obs_dim

    for aid in agent_ids:
        obs = obs_dict[aid]
        for k, v in enumerate(obs):
            if v != PREY:
                continue
            r, c = divmod(k, size)
            dx = c - center
            dy = r - center
            md = abs(dx) + abs(dy)
            if md < best_md:
                best_md = md
                best_agent = aid
                best_dx, best_dy = dx, dy

    return best_agent, best_dx, best_dy


def choose_leader_action(state, leader_id):
    obs = state.obs[leader_id]
    action = action_from_obs(obs, state.obs_dim)
    if action is None:
        action = DO_NOTHING
    return action


def choose_helper_action(state, helper_id, leader_action):
    """
    Helper policy given:
      - its own local obs
      - the leader's chosen action
    """
    obs_dim = state.obs_dim
    obs = state.obs[helper_id]
    legal_moves = legal_moves_from_obs(obs, obs_dim)

    if not legal_moves:
        return DO_NOTHING

    rngs = getattr(state, "rngs", {})
    rng = rngs.get(helper_id, random)

    # If helper also sees prey, chase greedily.
    if PREY in obs:
        a = action_from_obs(obs, obs_dim)
        if a in legal_moves:
            return a
        return rng.choice(legal_moves)

    # Otherwise, try to align with leader.
    if leader_action in legal_moves and leader_action != DO_NOTHING:
        return leader_action

    # Try an orthogonal move to create a pincer effect.
    if leader_action in (LEFT, RIGHT):
        preferred = [UP, DOWN]
    elif leader_action in (UP, DOWN):
        preferred = [LEFT, RIGHT]
    else:
        preferred = []

    candidates = [a for a in legal_moves if a in preferred]
    if candidates:
        return rng.choice(candidates)

    # Final fallback: any legal move.
    return rng.choice(legal_moves)


def choose_patrol_action(state, agent_id):
    """
    Shared patrol logic for use in both single-agent and joint methods.
    Uses unified state conventions: state.prev_actions, state.keep_prev_action, state.rngs.
    """
    obs_dim = state.obs_dim
    obs = state.obs[agent_id]
    legal_moves = legal_moves_from_obs(obs, obs_dim)

    if not legal_moves:
        return DO_NOTHING

    prev_actions = getattr(state, "prev_actions", {})
    keep_prev = bool(getattr(state, "keep_prev_action", True))
    rngs = getattr(state, "rngs", {})
    rng = rngs.get(agent_id, random)

    prev = prev_actions.get(agent_id, DO_NOTHING)

    if not keep_prev and prev in legal_moves:
        filtered = [a for a in legal_moves if a != prev]
        if filtered:
            legal_moves = filtered

    if keep_prev and prev in legal_moves and prev != DO_NOTHING:
        return prev

    return rng.choice(legal_moves)