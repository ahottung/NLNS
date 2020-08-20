from multiprocessing import Pool, Manager
from vrp.data_utils import read_instance
from copy import deepcopy
import numpy as np
import torch
import repair
import time
import math
import search


def lns_single_seach_job(args):
    try:
        id, config, instance_path, model_path, queue_jobs, queue_results, pkl_instance_id = args

        operator_pairs = search.load_operator_pairs(model_path, config)
        instance = read_instance(instance_path, pkl_instance_id)

        T_min = config.lns_t_min

        # Repeat until the process is terminated
        while True:
            solution, incumbent_cost = queue_jobs.get()
            incumbent_solution = deepcopy(solution)
            cur_cost = np.inf
            instance.solution = solution
            start_time_reheating = time.time()

            # Create a batch of copies of the same instances that can be repaired in parallel
            instance_copies = [deepcopy(instance) for _ in range(config.lns_batch_size)]

            iter = -1
            # Repeat until the time limit of one reheating iteration is reached
            while time.time() - start_time_reheating < config.lns_timelimit / config.lns_reheating_nb:
                iter += 1

                # Set the first config.lns_Z_param percent of the instances/solutions in the batch
                # to the last accepted solution
                for i in range(int(config.lns_Z_param * config.lns_batch_size)):
                    instance_copies[i] = deepcopy(instance)

                # Select an LNS operator pair (destroy + repair operator)
                selected_operator_pair_id = np.random.randint(0, len(operator_pairs))
                actor = operator_pairs[selected_operator_pair_id].model
                destroy_procedure = operator_pairs[selected_operator_pair_id].destroy_procedure
                p_destruction = operator_pairs[selected_operator_pair_id].p_destruction

                # Destroy instances
                search.destroy_instances(instance_copies, destroy_procedure, p_destruction)

                # Repair instances
                for i in range(int(len(instance_copies) / config.lns_batch_size)):
                    with torch.no_grad():
                        repair.repair(
                            instance_copies[i * config.lns_batch_size: (i + 1) * config.lns_batch_size], actor, config)

                costs = [instance.get_costs_memory(config.round_distances) for instance in instance_copies]

                # Calculate the T_max and T_factor values for simulated annealing in the first iteration
                if iter == 0:
                    q75, q25 = np.percentile(costs, [75, 25])
                    T_max = q75 - q25
                    T_factor = -math.log(T_max / T_min)
                    #print("tmax", T_max)

                min_costs = min(costs)

                # Update incumbent if a new best solution is found
                if min_costs <= incumbent_cost:
                    incumbent_solution = deepcopy(instance_copies[np.argmin(costs)].solution)
                    incumbent_cost = min_costs

                # Calculate simulated annealing temperature
                T = T_max * math.exp(
                    T_factor * (time.time() - start_time_reheating) / (config.lns_timelimit / config.lns_reheating_nb))

                # Accept a solution if the acceptance criteria is fulfilled
                if min_costs <= cur_cost or np.random.rand() < math.exp(-(min(costs) - cur_cost) / T):
                    instance.solution = instance_copies[np.argmin(costs)].solution
                    cur_cost = min_costs

            queue_results.put([incumbent_solution, incumbent_cost])

    except Exception as e:
        print("Exception in lns_single_search job: {0}".format(e))


def lns_single_search_mp(instance_path, timelimit, config, model_path, pkl_instance_id=None):
    instance = read_instance(instance_path, pkl_instance_id)
    start_time = time.time()
    instance.create_initial_solution()
    incumbent_costs = instance.get_costs(config.round_distances)
    instance.verify_solution(config)

    m = Manager()
    queue_jobs = m.Queue()
    queue_results = m.Queue()
    pool = Pool(processes=config.lns_nb_cpus)
    pool.map_async(lns_single_seach_job,
                   [(i, config, instance_path, model_path, queue_jobs, queue_results, pkl_instance_id) for i in
                    range(config.lns_nb_cpus)])
    # Distribute starting solution to search processes
    for i in range(config.lns_nb_cpus):
        queue_jobs.put([instance.solution, incumbent_costs])

    while time.time() - start_time < timelimit:
        # Receive the incumbent solution from a finished search process (reheating iteration finished)
        result = queue_results.get()
        if result != 0:
            if result[1] < incumbent_costs:
                incumbent_costs = result[1]
                instance.solution = result[0]
                print('incumbent_costs', incumbent_costs)
        # Distribute incumbent solution to search processes
        queue_jobs.put([instance.solution, incumbent_costs])

    pool.terminate()
    duration = time.time() - start_time
    instance.verify_solution(config)
    return instance.get_costs(config.round_distances), duration
