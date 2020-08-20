import torch
import numpy as np
import repair
import time
import multiprocessing as mp
import search
from vrp.data_utils import create_dataset, read_instances_pkl

EMA_ALPHA = 0.2


def lns_batch_search(instances, max_iterations, timelimit, operator_pairs, config):
    if len(instances) % config.lns_batch_size != 0:
        raise Exception("Instance set size must be multiple of lns_batch_size for batch search.")

    costs = [instance.get_costs_memory(config.round_distances) for instance in instances]  # Costs for each instance
    performance_EMA = [np.inf] * len(
        operator_pairs)  # Exponential moving average of avg. improvement in last iterations

    start_time = time.time()
    for iteration_id in range(max_iterations):

        if time.time() - start_time > timelimit:
            break

        mean_cost_before_iteration = np.mean(costs)

        solution_copies = [instance.get_solution_copy() for instance in instances]

        # Select an LNS operator pair (destroy + repair operator)
        if config.lns_adaptive_search:
            selected_operator_pair_id = np.argmax(performance_EMA)  # select operator pair with the best EMA
        else:
            selected_operator_pair_id = np.random.randint(0, len(operator_pairs))  # select operator pair at random
        actor = operator_pairs[selected_operator_pair_id].model
        destroy_procedure = operator_pairs[selected_operator_pair_id].destroy_procedure
        p_destruction = operator_pairs[selected_operator_pair_id].p_destruction

        start_time_destroy = time.time()

        # Destroy instances
        search.destroy_instances(instances, destroy_procedure, p_destruction)

        # Repair instances
        for i in range(int(len(instances) / config.lns_batch_size)):
            with torch.no_grad():
                repair.repair(instances[i * config.lns_batch_size: (i + 1) * config.lns_batch_size], actor,
                              config)

        destroy_repair_duration = time.time() - start_time_destroy

        for i in range(len(instances)):
            cost = instances[i].get_costs_memory(config.round_distances)
            # Only "accept" improving solutions
            if costs[i] < cost:
                instances[i].solution = solution_copies[i]
            else:
                costs[i] = cost

        # If adaptive search is used, update performance scores
        if config.lns_adaptive_search:
            delta = (mean_cost_before_iteration - np.mean(costs)) / destroy_repair_duration
            if performance_EMA[selected_operator_pair_id] == np.inf:
                performance_EMA[selected_operator_pair_id] = delta
            performance_EMA[selected_operator_pair_id] = performance_EMA[selected_operator_pair_id] * (
                        1 - EMA_ALPHA) + delta * EMA_ALPHA
           # print(performance_EMA)

    # Verify solutions
    for instance in instances:
        instance.verify_solution(config)

    return costs, iteration_id


def _lns_batch_search_job(args):
    (i, test_size, config, model_path) = args
    if config.instance_path is None:
        instances = create_dataset(size=test_size, config=config, seed=config.validation_seed + 1 + i)
    else:
        instances = read_instances_pkl(config.instance_path, test_size * i, test_size)

    lns_operations = search.load_operator_pairs(model_path, config)

    for instance in instances:
        instance.create_initial_solution()

    costs, nb_iterations = lns_batch_search(instances, config.lns_max_iterations, config.lns_timelimit, lns_operations,
                                            config)

    return i, costs, nb_iterations


def lns_batch_search_mp(config, model_path):
    if config.instance_path is None:
        nb_instances = config.test_size
    else:
        nb_instances = len(read_instances_pkl(config.instance_path))
    assert nb_instances % config.lns_nb_cpus == 0
    test_size_per_cpu = nb_instances // config.lns_nb_cpus

    if config.lns_nb_cpus > 1:
        with mp.Pool(config.lns_nb_cpus) as pool:
            results = pool.map(
                _lns_batch_search_job,
                [(i, test_size_per_cpu, config, model_path) for i in range(config.lns_nb_cpus)]
            )
    else:
        results = _lns_batch_search_job((0, test_size_per_cpu, config, model_path))
        results = [results]
    return results
