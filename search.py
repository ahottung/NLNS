import logging
import numpy as np
import os
import time
import torch
import search_single
from vrp.data_utils import read_instances_pkl
import glob
import search_batch
from actor import VrpActorModel


class LnsOperatorPair:

    def __init__(self, model, destroy_procedure, p_destruction):
        self.model = model
        self.destroy_procedure = destroy_procedure
        self.p_destruction = p_destruction


def destroy_instances(instances, destroy_procedure=None, destruction_p=None):
    for instance in instances:
        if destroy_procedure == "R":
            instance.destroy_random(destruction_p)
        elif destroy_procedure == "P":
            instance.destroy_point_based(destruction_p)
        elif destroy_procedure == "T":
            instance.destroy_tour_based(destruction_p)


def load_operator_pairs(path, config):
    if path.endswith('.pt'):
        model_paths = [path]
    else:
        model_paths = glob.glob(os.path.join(path, '*.pt'))

    if not model_paths:
        raise Exception(f"No operators found in {path}")

    lns_operator_pairs = []
    for model_path in model_paths:
        model_data = torch.load(model_path, config.device)

        actor = VrpActorModel(config.device, hidden_size=config.pointer_hidden_size).to(
            config.device)
        actor.load_state_dict(model_data['parameters'])
        actor.eval()

        operator_pair = LnsOperatorPair(actor, model_data['destroy_operation'], model_data['p_destruction'])
        lns_operator_pairs.append(operator_pair)
    return lns_operator_pairs


def evaluate_batch_search(config, model_path):
    assert model_path is not None, 'No model path given'

    logging.info('### Batch Search ###')
    logging.info('Starting search')
    start_time = time.time()

    results = search_batch.lns_batch_search_mp(config, model_path)

    runtime = (time.time() - start_time)
    instance_id, costs, iterations = [], [], []
    for r in results:
        instance_id.extend(list(range(len(r[1]) * r[0], len(r[1]) * (r[0] + 1))))
        costs.extend(r[1])
        iterations.append(r[2])

    path = os.path.join(config.output_path, "search", 'results.txt')
    np.savetxt(path, np.column_stack((instance_id, costs)), delimiter=',', fmt=['%i', '%f'])
    logging.info(
        f"Test set costs: {np.mean(costs):.3f} Total Runtime (s): {runtime:.1f} Iterations: {np.mean(iterations):.1f}")


def evaluate_single_search(config, model_path, instance_path):
    assert model_path is not None, 'No model path given'
    assert instance_path is not None, 'No instance path given'

    instance_names, costs, durations = [], [], []
    logging.info("### Single instance search ###")

    if instance_path.endswith(".vrp") or instance_path.endswith(".sd"):
        logging.info("Starting solving a single instance")
        instance_files_path = [instance_path]
    elif instance_path.endswith(".pkl"):
        instance_files_path = [instance_path] * len(read_instances_pkl(instance_path))
        logging.info("Starting solving a .pkl instance set")
    elif os.path.isdir(instance_path):
        instance_files_path = [os.path.join(instance_path, f) for f in os.listdir(instance_path)]
        logging.info("Starting solving all instances in directory")
    else:
        raise Exception("Unknown instance file format.")

    for i, instance_path in enumerate(instance_files_path):
        if instance_path.endswith(".pkl") or instance_path.endswith(".vrp") or instance_path.endswith(".sd"):
            for _ in range(config.nb_runs):
                cost, duration = search_single.lns_single_search_mp(instance_path, config.lns_timelimit, config,
                                                                    model_path, i)
                instance_names.append(instance_path)
                costs.append(cost)
                durations.append(duration)

    output_path = os.path.join(config.output_path, "search", 'results.txt')
    results = np.array(list(zip(instance_names, costs, durations)))
    np.savetxt(output_path, results, delimiter=',', fmt=['%s', '%s', '%s'], header="name, cost, runtime")

    logging.info(
        f"NLNS single search evaluation results: Total Nb. Runs: {len(costs)}, "
        f"Mean Costs: {np.mean(costs):.3f} Mean Runtime (s): {np.mean(durations):.1f}")
