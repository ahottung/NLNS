import numpy as np
from vrp.vrp_problem import VRPInstance
import pickle


class InstanceBlueprint:
    """Describes the properties of a certain instance type (e.g. number of customers)."""

    def __init__(self, nb_customers, depot_position, customer_position, nb_customer_cluster, demand_type, demand_min,
                 demand_max, capacity, grid_size):
        self.nb_customers = nb_customers
        self.depot_position = depot_position
        self.customer_position = customer_position
        self.nb_customers_cluster = nb_customer_cluster
        self.demand_type = demand_type
        self.demand_min = demand_min
        self.demand_max = demand_max
        self.capacity = capacity
        self.grid_size = grid_size


def get_blueprint(blueprint_name):
    type = blueprint_name.split('_')[0]
    instance = blueprint_name.split('_')[1]
    if type == "ALTR":
        import vrp.dataset_blueprints.ALTR
        return vrp.dataset_blueprints.ALTR.dataset[instance]
    elif type == "XE":
        import vrp.dataset_blueprints.XE
        return vrp.dataset_blueprints.XE.dataset[instance]
    elif type == "S":
        import vrp.dataset_blueprints.S
        return vrp.dataset_blueprints.S.dataset[instance]
    raise Exception('Unknown blueprint instance')


def create_dataset(size, config, seed=None, create_solution=False, use_cost_memory=True):
    instances = []
    blueprints = get_blueprint(config.instance_blueprint)

    if seed is not None:
        np.random.seed(seed)
    for i in range(size):
        if isinstance(blueprints, list):
            blueprint_rnd_idx = np.random.randint(0, len(blueprints), 1).item()
            vrp_instance = generate_Instance(blueprints[blueprint_rnd_idx], use_cost_memory)
        else:
            vrp_instance = generate_Instance(blueprints, use_cost_memory)
        instances.append(vrp_instance)
        if create_solution:
            vrp_instance.create_initial_solution()
    return instances


def generate_Instance(blueprint, use_cost_memory):
    depot_position = get_depot_position(blueprint)
    customer_position = get_customer_position(blueprint)
    demand = get_customer_demand(blueprint, customer_position)
    original_locations = np.insert(customer_position, 0, depot_position, axis=0)
    demand = np.insert(demand, 0, 0, axis=0)

    if blueprint.grid_size == 1000:
        locations = original_locations / 1000
    elif blueprint.grid_size == 1000000:
        locations = original_locations / 1000000
    else:
        assert blueprint.grid_size == 1
        locations = original_locations

    vrp_instance = VRPInstance(blueprint.nb_customers, locations, original_locations, demand, blueprint.capacity,
                               use_cost_memory)
    return vrp_instance


def get_depot_position(blueprint):
    if blueprint.depot_position == 'R':
        if blueprint.grid_size == 1:
            return np.random.uniform(size=(1, 2))
        elif blueprint.grid_size == 1000:
            return np.random.randint(0, 1001, 2)
        elif blueprint.grid_size == 1000000:
            return np.random.randint(0, 1000001, 2)
    elif blueprint.depot_position == 'C':
        if blueprint.grid_size == 1:
            return np.array([0.5, 0.5])
        elif blueprint.grid_size == 1000:
            return np.array([500, 500])
    elif blueprint.depot_position == 'E':
        return np.array([0, 0])
    else:
        raise Exception("Unknown depot position")


def get_customer_position_clustered(nb_customers, blueprint):
    assert blueprint.grid_size == 1000
    random_centers = np.random.randint(0, 1001, (blueprint.nb_customers_cluster, 2))
    customer_positions = []
    while len(customer_positions) + blueprint.nb_customers_cluster < nb_customers:
        random_point = np.random.randint(0, 1001, (1, 2))
        a = random_centers
        b = np.repeat(random_point, blueprint.nb_customers_cluster, axis=0)
        distances = np.sqrt(np.sum((a - b) ** 2, axis=1))
        acceptance_prob = np.sum(np.exp(-distances / 40))
        if acceptance_prob > np.random.rand():
            customer_positions.append(random_point[0])
    return np.concatenate((random_centers, np.array(customer_positions)), axis=0)


def get_customer_position(blueprint):
    if blueprint.customer_position == 'R':
        if blueprint.grid_size == 1:
            return np.random.uniform(size=(blueprint.nb_customers, 2))
        elif blueprint.grid_size == 1000:
            return np.random.randint(0, 1001, (blueprint.nb_customers, 2))
        elif blueprint.grid_size == 1000000:
            return np.random.randint(0, 1000001, (blueprint.nb_customers, 2))
    elif blueprint.customer_position == 'C':
        return get_customer_position_clustered(blueprint.nb_customers, blueprint)
    elif blueprint.customer_position == 'RC':
        customer_position = get_customer_position_clustered(int(blueprint.nb_customers / 2), blueprint)
        customer_position_2 = np.random.randint(0, 1001, (blueprint.nb_customers - len(customer_position), 2))
        return np.concatenate((customer_position, customer_position_2), axis=0)


def get_customer_demand(blueprint, customer_position):
    if blueprint.demand_type == 'inter':
        return np.random.randint(blueprint.demand_min, blueprint.demand_max + 1, size=blueprint.nb_customers)
    elif blueprint.demand_type == 'U':
        return np.ones(blueprint.nb_customers, dtype=int)
    elif blueprint.demand_type == 'SL':
        small_demands_nb = int(np.random.uniform(0.7, 0.95, 1).item() * blueprint.nb_customers)
        demands_small = np.random.randint(1, 11, size=small_demands_nb)
        demands_large = np.random.randint(50, 101, size=blueprint.nb_customers - small_demands_nb)
        demands = np.concatenate((demands_small, demands_large), axis=0)
        np.random.shuffle(demands)
        return demands
    elif blueprint.demand_type == 'Q':
        assert blueprint.grid_size == 1000
        demands = np.zeros(blueprint.nb_customers, dtype=int)
        for i in range(blueprint.nb_customers):
            if (customer_position[i][0] > 500 and customer_position[i][1] > 500) or (
                    customer_position[i][0] < 500 and customer_position[i][1] < 500):
                demands[i] = np.random.randint(51, 101, 1).item()
            else:
                demands[i] = np.random.randint(1, 51, 1).item()
        return demands
    elif blueprint.demand_type == 'minOrMax':
        demands_small = np.repeat(blueprint.demand_min, blueprint.nb_customers * 0.5)
        demands_large = np.repeat(blueprint.demand_max, blueprint.nb_customers - (blueprint.nb_customers * 0.5))
        demands = np.concatenate((demands_small, demands_large), axis=0)
        np.random.shuffle(demands)
        return demands
    else:
        raise Exception("Unknown customer demand.")


def read_instance(path, pkl_instance_idx=0):
    if path.endswith('.vrp'):
        return read_instance_vrp(path)
    elif path.endswith('.sd'):
        return read_instance_sd(path)
    elif path.endswith('.pkl'):
        return read_instances_pkl(path, pkl_instance_idx, 1)[0]
    else:
        raise Exception("Unknown instance file type.")


def read_instance_vrp(path):
    file = open(path, "r")
    lines = [ll.strip() for ll in file]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension
        elif line.startswith('DEMAND_SECTION'):
            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension

        i += 1

    original_locations = locations[:, 1:]
    locations = original_locations / 1000
    demand = demand[:, 1:].squeeze()

    instance = VRPInstance(dimension - 1, locations, original_locations, demand, capacity)
    return instance


def read_instance_sd(path):
    file = open(path, "r")
    lines = [ll.strip() for ll in file]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension
        elif line.startswith('DEMAND_SECTION'):
            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension

        i += 1

    original_locations = locations[:, 1:]
    locations = original_locations / (original_locations[0, 0] * 2)
    demand = demand[:, 1:].squeeze()

    instance = VRPInstance(dimension - 1, locations, original_locations, demand, capacity)
    return instance


def read_instances_pkl(path, offset=0, num_samples=None):
    instances = []

    with open(path, 'rb') as f:
        data = pickle.load(f)

    if num_samples is None:
        num_samples = len(data)

    for args in data[offset:offset + num_samples]:
        depot, loc, demand, capacity, *args = args
        loc.insert(0, depot)
        demand.insert(0, 0)

        locations = np.array(loc)
        demand = np.array(demand)

        instance = VRPInstance(len(loc) - 1, locations, locations, demand, capacity)
        instances.append(instance)

    return instances
