
import time
import inspect
from pprint import pprint
import argparse
import random

import posggym
# actions: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT (per this env)
# obs cells: 0=EMPTY, 1=WALL, 2=PREDATOR, 3=PREY

import posggym.envs.grid_world.predator_prey as pp

from gymnasium.wrappers import RecordVideo

from wrappers import ActionLoggingWrapper
from observers import MinimalObserver

import gtpyhop
from pp_htn import (
    domain_name, DO_NOTHING, UP, DOWN, LEFT, RIGHT, PRED, PREY, EMPTY
)

from plot_utils import plot_trajectories, record_positions

#KEEP_PREV_ACTION = True  # whether to prefer continuing in same direction when patrolling (planner uses this)


#print(pp.__file__)

ACTION_NAMES = {
    0: "STAY",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
}

def plan_to_actions(plan):
    """Plan should be in the list[tuple] for: [('do', 0, 4)] --> agent 0 do action 4"""
    if not plan:
        return DO_NOTHING
    op, aid, act = plan[0]
    return int(act)
    # for op, aid, act in plan:
    #     return act
    

def my_policy(obs, agent_id):
    """A very simple policy for testing purposes."""
    # Move up
    return 1

def build_planner_state(env, observations):
    """Build a GTPyhop state from the current environment observations."""
    s = gtpyhop.State("tick")
    s.obs = {agent_id: observations[agent_id] for agent_id in env.agents }
    s.obs_dim = env.unwrapped.model.obs_dim
    
    return s

def main():
    """
    Run POSGGym Predator-Prey with GTPyhop HTN planner.
    
    Capture Criteria:
    Prey are captured when at least prey_strength predators are in adjacent cells, 
    where 1 <= prey_strength <= min(4, num_predators).
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run POSGGym Predator-Prey with GTPyhop HTN planner.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging.")
    parser.add_argument("--keep-prev-action", dest="keep_prev_action", action="store_true",
                        help="When patrolling, allow repeating the previous action.")
    parser.add_argument("--no-keep-prev-action", dest="keep_prev_action", action="store_false",
                        help="When patrolling, exclude the previous action if alternatives exist.")
    parser.add_argument("--time-horizon", type=int, default=200, help="Maximum number of steps per episode.")
    
    parser.set_defaults(keep_prev_action=True)
    args = parser.parse_args()
    
    debug = args.debug
    KEEP_PREV_ACTION = args.keep_prev_action
    #print("Testing environment creation...")
    
    # Environment parameters
    time_horizon=args.time_horizon
    TARGET_FPS = 5
    SLEEP = 1.0 / TARGET_FPS
    
    
    """Create POSGGym Predator-Prey environment.
    prey_strength - how many predators are required to capture each prey,
    minimum is 1 and maximum is min(4, num_predators). 
    If None this is set to min(4, num_predators) (default = ‘None`)
    
    Note: if time_horizon is > max_episode_steps, env will terminate early at max_episode_steps
    """
    env = posggym.make(
        "PredatorPrey-v0",
        max_episode_steps=200,
        grid="10x10",
        num_predators=2,
        num_prey=1,
        render_mode= "human" ,
    )
    # Instantiate environment with action logging wrapper that has more detailed logging
    env = ActionLoggingWrapper(env, debug=debug)
    #env = RecordVideo(env, video_folder="./videos/", name_prefix="pred_prey", episode_trigger=lambda x: True)
   
    if debug:
        print(f"Run POSGGym Predator-Prey with GTPyhop HTN planner. [DEBUG MODE]")
        print(f"[DEBUG] Printing GTPyhop Domain")
        gtpyhop.print_domain()
    
    
    observations, infos = env.reset(seed=42)
    
    position_history = {"predators" : {}, "prey" : {} }
    record_positions(env, position_history, init=True)
    
        
    # Per-agent persistent memory lives OUTSIDE GTPyhop/state
    agent_ids = list(env.agents)
    agent_memory = {
        aid: {
            "rng": random.Random(100 + i),
            "prev_action": DO_NOTHING,
            "last_seen_prey": None,
        }
        for i, aid in enumerate(agent_ids)
    }
    
    if debug:
        print("=========================")
        print(f"[DEBUG] Starting episode with agents: {env.agents}")
        print("[DEBUG] env.agents:", list(env.agents))
        print("[DEBUG] obs keys:  ", list(observations.keys()))
        for aid in env.agents:
            print(aid, "action space:", env.action_spaces[aid])
        print("=========================")
        
        
    observer = MinimalObserver(pretty=False)
    observer.on_reset(env, observations,infos)
    
    
    

    for t in range(time_horizon):
        # Build a GTPyhop state with exactly the methods we need
        s = build_planner_state(env, observations)
        
        # 
        actions = {}
        
        # Controller calls planner for each agent 
        for agent_id in env.agents:
            s.rng = agent_memory[agent_id]["rng"]
            # Inject memory + flags into state for planner use
            s.prev_action = agent_memory[agent_id]["prev_action"]
            s.keep_prev_action = KEEP_PREV_ACTION

           
            plan = gtpyhop.find_plan(s, [("choose_action", agent_id)])
            actions[agent_id] = plan_to_actions(plan)
            
        
        if debug:
            readable = {aid: f"{act} ({ACTION_NAMES[act]})" for aid, act in actions.items()}
            print("[DEBUG] Actions:")
            pprint(readable)
            print("[DEBUG] Raw:", actions)
            print("=========================")
       

        # step environment
        observations, rewards, terminations, truncations, all_done, infos = env.step(actions)
        
        # Record all positions for prey and predators for plotting
        record_positions(env, position_history)
        
        observer.on_step(t, observations, rewards, terminations, truncations, infos)
        
        # Persist last executed action
        for aid in env.agents:
            agent_memory[aid]["prev_action"] = actions[aid]
        
        
        # 4) compact tick summary
        if debug:
            pprint(
                f"[DEBUG] [t={t}] | obs={observations} "
                f"| rewards={len(rewards)} | done={all_done} "
                f"| term={sum(1 for v in terminations.values() if v)} "
                f"| trunc={sum(1 for v in truncations.values() if v)}"
            )

        env.render() #not needed during video recording
        #time.sleep(0.03)
        time.sleep(SLEEP)

        if all_done:
            reason = "time_limit" if any(truncations.values()) else "task_solved"
            #print(f"Episode ended at t={t} due to {reason}")
            observer.on_episode_end(reason)
            break

    print(f"[INFO] Episode finished after {t} steps")
    
    grid_size = env.unwrapped.model.grid_size if hasattr(env.unwrapped.model, "grid_size") else (10, 10)
    env.close()
    
    plot_trajectories(position_history, grid_size, save_path="figures/final_positions.png")
    print("[INFO] Saved trajectory plot to figures/final_positions.png")
    


if __name__ == "__main__":
    main()


    