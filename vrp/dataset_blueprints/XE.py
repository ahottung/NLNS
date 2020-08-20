from vrp.data_utils import InstanceBlueprint

dataset = {}
dataset['1'] = InstanceBlueprint(nb_customers=100, depot_position='R', customer_position='RC', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=100, capacity=206, grid_size=1000)  # based on instance X 1 of the paper
# "New Benchmark Instances for the Capacitated Vehicle Routing Problem"
dataset['2'] = InstanceBlueprint(nb_customers=124, depot_position='R', customer_position='C', nb_customer_cluster=5,
    demand_type='Q', demand_min=None, demand_max=None, capacity=188, grid_size=1000)  # based on instance X 6
dataset['3'] = InstanceBlueprint(nb_customers=128, depot_position='E', customer_position='RC', nb_customer_cluster=8,
    demand_type='inter', demand_min=1, demand_max=10, capacity=39, grid_size=1000)  # based on instance X 7
dataset['4'] = InstanceBlueprint(nb_customers=161, depot_position='C', customer_position='RC', nb_customer_cluster=8,
    demand_type='inter', demand_min=50, demand_max=100, capacity=1174, grid_size=1000)  # based on instance X 14
dataset['5'] = InstanceBlueprint(nb_customers=180, depot_position='R', customer_position='C', nb_customer_cluster=6,
    demand_type='U', demand_min=None, demand_max=None, capacity=8, grid_size=1000)  # based on instance X 18
dataset['6'] = InstanceBlueprint(nb_customers=185, depot_position='R', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=50, demand_max=100, capacity=974, grid_size=1000)  # based on instance X 19
dataset['7'] = InstanceBlueprint(nb_customers=199, depot_position='R', customer_position='C', nb_customer_cluster=8,
    demand_type='Q', demand_min=None, demand_max=None, capacity=402, grid_size=1000)  # based on instance X 22
dataset['8'] = InstanceBlueprint(nb_customers=203, depot_position='C', customer_position='RC', nb_customer_cluster=6,
    demand_type='inter', demand_min=50, demand_max=100, capacity=836, grid_size=1000)  # based on instance X 23
dataset['9'] = InstanceBlueprint(nb_customers=213, depot_position='C', customer_position='C', nb_customer_cluster=4,
    demand_type='inter', demand_min=1, demand_max=100, capacity=944, grid_size=1000)  # based on instance X 25
dataset['10'] = InstanceBlueprint(nb_customers=218, depot_position='E', customer_position='R', nb_customer_cluster=None,
    demand_type='U', demand_min=None, demand_max=None, capacity=3, grid_size=1000)  # based on instance X 26
dataset['11'] = InstanceBlueprint(nb_customers=236, depot_position='E', customer_position='R', nb_customer_cluster=None,
    demand_type='U', demand_min=None, demand_max=None, capacity=18, grid_size=1000)  # based on instance X 30
dataset['12'] = InstanceBlueprint(nb_customers=241, depot_position='E', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=1, demand_max=10, capacity=28, grid_size=1000)  # based on instance X 31
dataset['13'] = InstanceBlueprint(nb_customers=269, depot_position='C', customer_position='RC', nb_customer_cluster=5,
    demand_type='inter', demand_min=50, demand_max=100, capacity=585, grid_size=1000)  # based on instance X 37
dataset['14'] = InstanceBlueprint(nb_customers=274, depot_position='R', customer_position='C', nb_customer_cluster=3,
    demand_type='U', demand_min=None, demand_max=None, capacity=10, grid_size=1000)  # based on instance X 38
dataset['15'] = InstanceBlueprint(nb_customers=279, depot_position='E', customer_position='R', nb_customer_cluster=None,
    demand_type='SL', demand_min=None, demand_max=None, capacity=192, grid_size=1000)  # based on instance X 39
dataset['16'] = InstanceBlueprint(nb_customers=293, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=1, demand_max=100, capacity=285, grid_size=1000)  # based on instance X 42
dataset['17'] = InstanceBlueprint(nb_customers=297, depot_position='R', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=1, demand_max=10, capacity=55, grid_size=1000)  # based on instance X 43
