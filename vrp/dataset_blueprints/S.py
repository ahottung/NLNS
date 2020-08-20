from vrp.data_utils import InstanceBlueprint

dataset = {}
dataset['76D1'] = InstanceBlueprint(nb_customers=75, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=2, demand_max=15, capacity=160, grid_size=1)
dataset['76D2'] = InstanceBlueprint(nb_customers=75, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=16, demand_max=47, capacity=160, grid_size=1)
dataset['76D3'] = InstanceBlueprint(nb_customers=75, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=16, demand_max=79, capacity=160, grid_size=1)
dataset['76D4'] = InstanceBlueprint(nb_customers=75, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=60, demand_max=143, capacity=160, grid_size=1)
dataset['101D1'] = InstanceBlueprint(nb_customers=100, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=2, demand_max=15, capacity=160, grid_size=1)
dataset['101D2'] = InstanceBlueprint(nb_customers=100, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=16, demand_max=47, capacity=160, grid_size=1)
dataset['101D3'] = InstanceBlueprint(nb_customers=100, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=16, demand_max=79, capacity=160, grid_size=1)
dataset['101D5'] = InstanceBlueprint(nb_customers=100, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=48, demand_max=143, capacity=160, grid_size=1)

dataset['101A1'] = InstanceBlueprint(nb_customers=100, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=2, demand_max=47, capacity=160, grid_size=1)
dataset['101A2'] = InstanceBlueprint(nb_customers=100, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=16, demand_max=143, capacity=160, grid_size=1)
dataset['101A3'] = InstanceBlueprint(nb_customers=100, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=2, demand_max=143, capacity=160, grid_size=1)






