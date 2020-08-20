import argparse
import os
import numpy as np
from vrp.data_utils import generate_Instance, get_blueprint


def write_vrplib(filename, loc, demand, capacity, grid_size, name="problem"):
    assert grid_size == 1000 or grid_size == 1000000

    with open(filename, 'w+') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "CVRP"),
                ("DIMENSION", len(loc)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", capacity)
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate(loc)
        ]))
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate(demand)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data')
    parser.add_argument('--instance_blueprint', default=None, type=str)
    parser.add_argument('--dataset_size', default=1, type=int)
    parser.add_argument('--seed', default=None, type=int)

    config = parser.parse_args()

    assert config.instance_blueprint.startswith("XE_"), 'Only XE are supported'

    np.random.seed(config.seed)
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)
    blueprint = get_blueprint(config.instance_blueprint)

    for i in range(config.dataset_size):
        instance = generate_Instance(blueprint, False)

        name = "{}_seed_{}_id_{}".format(config.instance_blueprint, config.seed, i)
        filename = os.path.join(config.data_dir, name + ".vrp")
        write_vrplib(filename, instance.original_locations, instance.demand, instance.capacity,
                     blueprint.grid_size, name)
