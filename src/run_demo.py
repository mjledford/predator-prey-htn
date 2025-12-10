
import time
import inspect
from pprint import pprint
import argparse
import random
import os
import posggym
# actions: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT (per this env)
# obs cells: 0=EMPTY, 1=WALL, 2=PREDATOR, 3=PREY

# seed = 42 for 

import posggym.envs.grid_world.predator_prey as pp

from gymnasium.wrappers import RecordVideo

from wrappers import ActionLoggingWrapper
from observers import MinimalObserver

from plan_utils import (
    plan_to_actions,
    joint_plan_to_actions,
    build_planner_state,
)


import pp_htn

import gtpyhop

from constants import (
    DO_NOTHING, UP, DOWN, LEFT, RIGHT,
    EMPTY, WALL, PRED, PREY,
    DIRS, ORDERED_DIRS, ACTION_NAMES, FIG_DIR
)

from plot_utils import plot_trajectories, record_positions, plot_capture_statistics, plot_avg_steps_for_k, plot_k_vs_steps, plot_comm_modes_comparison, plot_comm_modes_success_rates, plot_k_vs_costs
import matplotlib.pyplot as plt

from comm_module import HTNCommModule, CommStats

from sweep_utils import sweep_k_sync, sweep_comm_modes

#print(pp.__file__)




def run_single_episode (
    run_idx: int,
    seed: int,
    time_horizon: int = 200,
    debug: bool = False,
    keep_prev_action: bool = True,
    render: bool = False,
    comm_mode: str = "full",
    k_sync: int = 5):
    """
    Run one Predator-Prey episode and return:
        captured (bool): whether prey was captured
        steps_to_capture (int or None): number of env steps until capture
                                        (None if not captured within horizon)
    """
    TARGET_FPS = 5
    SLEEP = 1.0 / TARGET_FPS
    
    
    """Create POSGGym Predator-Prey environment.
    prey_strength - how many predators are required to capture each prey,
    minimum is 1 and maximum is min(4, num_predators). 
    If None this is set to min(4, num_predators) (default = ‘None`)
    
    Note: if time_horizon is > max_episode_steps, env will terminate early at max_episode_steps
    """
    save_plot_trajectories_each_episode = False
    env = posggym.make (
        "PredatorPrey-v0",
        max_episode_steps=time_horizon,  # keep aligned with horizon
        grid="10x10",
        num_predators=2,
        num_prey=1,
        render_mode="human" if render else None,
    )
    # Instantiate environment with action logging wrapper that has more detailed logging
    env = ActionLoggingWrapper(env, debug=debug)
    #env = RecordVideo(env, video_folder="./videos/", name_prefix="pred_prey", episode_trigger=lambda x: True)
   
    if debug:
        print(f"Run POSGGym Predator-Prey with GTPyhop HTN planner. [DEBUG MODE]")
        print(f"[DEBUG | Run_IDX={run_idx}] Printing GTPyhop Domain")
        gtpyhop.print_domain()
    
    # seed = 42 for reproducible run where the prey is captured around cell (10,9)
    # seed = 43 is a run where the agents get stuck in the top right and don't move
    observations, infos = env.reset(seed=seed)
    captured = False
    steps_to_capture = None
    all_done = False
    
    position_history = {"predators" : {}, "prey" : {} }
    record_positions(env, position_history, init=True)
    
        
    # Per-agent persistent memory lives OUTSIDE GTPyhop/state
    agent_ids = list(env.agents)
    agent_memory = {
        aid: {
            "rng": random.Random(seed * 1000 + i),
            "prev_action": DO_NOTHING,
            "last_seen_prey": None,
        }
        for i, aid in enumerate(agent_ids)
    }
    
    controller = HTNCommModule(mode=comm_mode, k_sync=k_sync, debug=debug)
    
    if debug:
        print("=========================")
        print(f"[DEBUG | Run_IDX={run_idx}] Starting episode with agents: {env.agents}")
        print("[DEBUG] env.agents:", list(env.agents))
        print("[DEBUG] obs keys:  ", list(observations.keys()))
        for aid in env.agents:
            print(aid, "action space:", env.action_spaces[aid])
        print("=========================")
        
        
    observer = MinimalObserver(pretty=False, debug=debug, run_idx=run_idx)
    observer.on_reset(env, observations,infos)
    
    
    

    for t in range(time_horizon):
        # Ask comm module to handle communication + planning + joint action
        actions = controller.decide_actions(
            t=t,
            env=env,
            observations=observations,
            agent_memory=agent_memory,
            keep_prev_action=keep_prev_action,
        )
        
       
        
        if debug:
            readable = {aid: f"{act} ({ACTION_NAMES[act]})" for aid, act in actions.items()}
            print("[DEBUG] Actions:", readable)
       

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
            print(f"[DEBUG] [t={t}] | done={all_done} | term={terminations} | trunc={truncations}")
        
        if render:
            env.render()
            time.sleep(SLEEP)

        if all_done:
            # Heuristic: capture => at least one True in terminations
            if any(terminations.values()):
                captured = True
                # t is 0-based index of this step, so steps taken = t+1
                steps_to_capture = t + 1
                reason = "task_solved"
            else:
                reason = "time_limit"
            observer.on_episode_end(reason)
            break
        
    if not all_done:
        # Horizon hit without env signalling all_done
        if any(terminations.values()):
            captured = True
            steps_to_capture = time_horizon
        else:
            captured = False
            steps_to_capture = None   

    print(f"[INFO] Episode finished after {t} steps: [SEED={seed}]")
    print(f"[INFO] Comm stats: messages={controller.stats.messages}, replans={controller.stats.replans}")

    
    grid_size = env.unwrapped.model.grid_size if hasattr(env.unwrapped.model, "grid_size") else (10, 10)
    env.close()
    
    plot_path = os.path.join(FIG_DIR, f"trajectories_seed_{seed}.png")
    if save_plot_trajectories_each_episode:
        plot_trajectories(position_history, grid_size, save_path=plot_path)
        print("[INFO] Saved trajectory plot to figures/trajectories_seed_{seed}.png")
    
    
    #plot_trajectories(position_history, grid_size, save_path=plot_path)
    
    
    return captured, steps_to_capture, controller.stats
    

def main():
    """
    Run multiple POSGGym Predator-Prey episodes with GTPyhop HTN planner,
    report average capture time, and plot per-run capture time.
    
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
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--render-last", action="store_false", help="Render the last episode visually.")
    parser.add_argument("--seed", type=int, default=None, help="Global experiment seed (optional). If not set, seeds vary per episode.")
    parser.add_argument("--rerun-seed", type=int, default=None, help="Run exactly one episode with this seed (overrides num-episodes and base seed).")
    parser.add_argument("--comm-mode", type=str, default="full", choices=["full", "periodic", "event", "none"], help="Communication mode between agents and planner.")
    parser.add_argument("--k-sync", type=int, default=5, help="Synchronization interval for periodic communication (comm-mode=periodic).")
    
    
    parser.set_defaults(keep_prev_action=True)
    args = parser.parse_args()
    
    debug = args.debug
    keep_prev_action = args.keep_prev_action
    time_horizon=args.time_horizon
    num_episodes=args.num_episodes
    comm_mode = args.comm_mode
    k_sync = args.k_sync
    
    # Data structures for metrics
    capture_times = []
    all_times = []
    successes=0
    total_messages = 0
    total_replans = 0
    
    # ---- Print configuration summary ----
    print("\n================ RUN CONFIG ================")
    print(f"Episodes:              {num_episodes}")
    print(f"Time horizon:          {time_horizon}")
    print(f"Grid:                  10x10")            # fixed 
    print(f"Predators:             2")                # fixed
    print(f"Prey:                  1")                # fixed
    print(f"Planner:               Joint HTN (choose_joint_action)")
    print(f"Keep previous action:  {keep_prev_action}")
    print(f"Debug mode:            {debug}")
    print(f"Render last episode:   {args.render_last}")
    print(f"Comm mode:            {comm_mode}")
    print(f"k_sync (periodic):    {k_sync}")
    print("============================================\n")
    
    # If rerun-seed is given, do that and exit early.
    if args.rerun_seed is not None:
        print(f"[INFO] Re-running single episode with seed {args.rerun_seed}")
        run_single_episode(
            run_idx=0,
            seed=args.rerun_seed,
            time_horizon=time_horizon,
            debug=debug,
            keep_prev_action=keep_prev_action,
            render=True,
            comm_mode=comm_mode,
            k_sync=k_sync
        )
        return

    # Otherwise, normal multi-episode run: set up base_seed
    if args.seed is not None:
        base_seed = args.seed
        print(f"[INFO] Using fixed base seed: {base_seed}")
    else:
        base_seed = random.randint(0, 10**6)
        print(f"[INFO] No seed provided. Using random base seed: {base_seed}")
    
    for run_idx in range(num_episodes):
        #seed = 42 + run_idx  # different seed per run
        seed = base_seed + run_idx
        
        render = args.render_last and (run_idx == num_episodes - 1)
        #render = True
        if debug:
            print(f"\n[INFO] === Run {run_idx+1}/{num_episodes}, seed={seed} ===")

        captured, steps, stats = run_single_episode(
            run_idx=run_idx,
            seed=seed,
            time_horizon=time_horizon,
            debug=debug,
            keep_prev_action=keep_prev_action,
            render=render,
            k_sync=k_sync,
            comm_mode=comm_mode,
        )
        total_messages += stats.messages
        total_replans += stats.replans
        
        if captured:
            successes += 1
            capture_times.append(steps)
        # For all_times, treat failures as horizon
        all_times.append(steps if steps is not None else time_horizon)

    # ---- Print stats ----
    print("\n================ RESULTS ================")
    print(f"Total runs:           {num_episodes}")
    print(f"Successful captures:  {successes}")
    print(f"Success rate:         {successes / num_episodes:.3f}")

    if capture_times:
        avg_capture_time = sum(capture_times) / len(capture_times)
        print(f"Avg steps to capture (successful episodes only): {avg_capture_time:.2f}")
    else:
        avg_capture_time = None
        print("No captures occurred; cannot compute average capture time.")

    avg_steps_all = sum(all_times) / len(all_times)
    avg_messages = total_messages / num_episodes
    avg_replans = total_replans / num_episodes
    print(f"Avg steps per episode (including failures):      {avg_steps_all:.2f}")
    print(f"Avg messages per episode:  {avg_messages:.2f}")
    print(f"Avg replans per episode:   {avg_replans:.2f}")
    print("=========================================\n")
    
    # ---- Call the centralized plotting function ----
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fig_dir = os.path.join(ROOT, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    plot_capture_statistics(
        all_times=all_times,
        capture_times=capture_times,
        avg_capture_time=avg_capture_time,
        avg_steps_all=avg_steps_all,
        save_dir=FIG_DIR,
    )
    
    if comm_mode == "periodic":
        plot_avg_steps_for_k(avg_capture_time, k_sync, save_dir=FIG_DIR)
    
    


if __name__ == "__main__":
    # UNCOMMENT TO DO K-SWEEP DIRECTLY FROM THIS FILE
    # k_values = [1, 5, 10, 20, 50]
    # results = sweep_k_sync(
    #     seed = 123456,
    #     k_values=k_values,
    #     num_episodes=20,
    #     time_horizon=200,
    #     debug=False,
    #     keep_prev_action=True,
    # )
    
    # plot_path = os.path.join(FIG_DIR, "k_vs_steps.png")
    # plot_k_vs_steps(results, save_path=plot_path, line=True)
    #***************************************************************
    
    # UNCOMMENT TO DO COMM-MODE SWEEP DIRECTLY FROM THIS FILE
    # results = sweep_comm_modes(
    #     seed=123456,
    #     num_episodes=20,
    #     time_horizon=200,
    #     debug=False,
    #     keep_prev_action=True,
    #     k_sync=10
    # )
    # plot_path = os.path.join(FIG_DIR, "comm_modes_vs_steps.png")
    # plot_comm_modes_comparison(results, save_path=plot_path)
    # plot_comm_modes_success_rates(results, save_path=os.path.join(FIG_DIR, "comm_modes_success_rates.png"))
    #***************************************************************
    
    # UNCOMMENT TO PLOT K-SWEEP VS COST
    # k_values = [1, 5, 10, 20, 50]
    # results = sweep_k_sync(
    #     seed = 123456,
    #     k_values=k_values,
    #     num_episodes=20,
    #     time_horizon=200,
    #     debug=False,
    #     keep_prev_action=True,
    # )
    # plot_path = os.path.join(FIG_DIR, "k_vs")
    # plot_k_vs_costs(results, save_path_prefix=plot_path)
    
    #***************************************************************
    # OTHERWISE, UNCOMMENT TO RUN MAIN EXPERIMENT FUNCTION
    main()


    