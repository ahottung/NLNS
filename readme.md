# NLNS - Neural Large Neighborhood Search

This repository contains the code used for the experiments in the paper "Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem" (https://arxiv.org/abs/1911.09539).

Neural large neighborhood search (NLNS) integrates learned heuristics in a large neighborhood search framework to solve the capacitated vehicle routing problem (CVRP) and the split delivery vehicle routing problem (SDVRP). During the search for solutions, NLNS repeditatly "destroys" routing problem solutions and then "repairs" them using deep neural network models with attention mechanism. The models are trained via reinforcement learning. In contrast to many existing machine learning based approaches for solving routing problems, NLNS is able to profit from longer runtimes (> 60s) and can be applied to real world-sized routing problem instances.

### Instances 

The XE instances used for the experiments in the paper can be found in the folder "instances". Each instance set consists of 20 instances that have similar properties. More instances with similar properties can be generated as described below.

### Paper
```
@inproceedings{
    hottung2020neural,
    title={Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem},
    author={Andr{\'e} Hottung and Kevin Tierney},
    booktitle={24th European Conference on Artificial Intelligence (ECAI 2020)},
    year={2020}
}
```

## Requirements

NLNS requires python (>= 3.6) and the following python packages:

- numpy
- pytorch (we used version 1.3.1 for evaluating NLNS)

## Quick Start

By default, NLNS uses the GPU. If you want to run NLNS on the CPU only, use the `--device cpu` option.

### Single Instance Search

To solve the provided single instance *XE_1_seed_123_id_0.vrp* (that is part of the XE_1 instance group) using the provided pre-trained models run the following command: 


```bash
python3 main.py --mode eval_single --model_path trained_models/cvrp/XE_1 --instance_path instances/XE_1/XE_1_seed_123_id_0.vrp --lns_nb_cpus 10 --round_distances
```

The search will take 180 seconds. All models in the directory `trained_models/cvrp/XE_1` are used during the search. The `--lns_nb_cpus option` flag can be used to define the number of used CPUs. The flag `--round_distances` enables rounding the distances between customers (which is usually done by non-ML based methods).

You can also solve all instances in a directory consecutively via single instance search, e.g., by using `--instance_path instances/XE_1`.

###  Batch Search

Batch search mode allows to quickly solve a set of instances in parallel. To solve the instance set *vrp20_test_seed1234.pkl* which contains 10.000 VRP instances with 20 customers and was generated using the code from [the attention model approach](https://github.com/wouterkool/attention-learn-to-route/) run:

```bash
python3 main.py --mode eval_batch --model_path trained_models/cvrp/altr_C_20 --lns_nb_cpus 10 --instance_path instances/ALTR/vrp20_test_seed1234.pkl --lns_batch_size 1000 --lns_timelimit 420 --lns_adaptive_search 
```

The search will take 420 seconds.

## Usage

By default, NLNS uses the GPU. If you want to run NLNS on the CPU only, use the `--device cpu` option.

### Single Instance Search

The single instance search uses simulated annealing (SA). If you want to solve instances that have significantly different objective function values than the provided instances you need to adjust (or better tune) the start and end temperatures of the SA mechanism (via `--lns_t_min` and `--lns_t_max`) . If you want to adjust the runtime (via `--lns_timelimit`) it is advised to also adjust the number of reheating iterations (`--lns_reheating_nb`) of the SA mechanism. For example, for the instance sets X_8 to X_17 we can give more time to NLNS using the following command:

```bash
python3 main.py --mode eval_single --model_path trained_models/cvrp/XE_8 --instance_path instances/XE_8 --lns_nb_cpus 10 --round_distances --lns_timelimit 600 --lns_reheating_nb 10
```

If you want to solve each instance more than once (for evaluation purposes) you can do that using the `--nb_runs` flag. The batch size of the search can be set via the `--nlns_batch_size` flag.

##### Split delivery vehicle routing problem

To allow split deliveries use the `--allow_split_delivery` flag. For example, to solve the provided instance *S51D1.sd* via single instance search run: 

```bash
python3 main.py --mode eval_single --model_path trained_models/sdvrp/S101A1 --instance_path instances/S/S51D1.sd --lns_nb_cpus 10 --lns_t_min 0.1 --round_distances --allow_split_delivery
```

### Batch Search

Batch search mode does not use a SA-based acceptance criteria. Optionally, the model selection probability can be adapted throughout the search based on the observed model performance (similar to adaptive large neighborhood search). This can be enabled by the `--lns_adaptive_search` flag.

##### Split delivery vehicle routing problem

 To solve the instances of the instance set *vrp20_test_seed1234.pkl* allowing split deliveries you can run:

```bash
python3 main.py --mode eval_batch --model_path trained_models/sdvrp/altr_SD_20 --lns_nb_cpus 10 --instance_path instances/ALTR/vrp20_test_seed1234.pkl --lns_batch_size 1000 --lns_timelimit 600 --lns_adaptive_search --allow_split_delivery
```

### Training

A model (i.e., a repair operator) is trained to repair instances from a known distribution that have been destroyed using a specific destroy operator. For example, run the following command to train a model that learns to repair instances from the *ALTR_20* instance group that have been destroyed using the tour-based destroy procedure (T) with a degree of destruction of 20%:

```
python3 main.py --mode train --instance_blueprint ALTR_20 --lns_destruction T --lns_destruction_p 0.2 --nb_batches_training_set 500 --lns_timelimit 300 --lns_batch_size 500
```

The size of the training set can be defined by the flag ``--nb_batches_training_set`` in batches (we use 1500 batches in our experiments). The batch size used training can be set via the `--batch_size` flag. The size of the validation and test set can be defined by the flags ``--valid_size`` and ``--test_size`` in absolute number of instances. The instances used for training and validation are generated before the training (which might take a while).  Every 5000 batches the model is validated in a batch search (with a batch size defined by `--lns_batch_size`). After the training the model is evaluated on the test set. The runtime of the batch search on the test set can be adjusted using the `--lns_timelimit` flag. For example, to train models for the *XE* instances we use `--lns_timelimit 3600`. The runtime of the validation search is adjusted automatically based on the validation set size.

The `--lns_destruction` flag defines the destroy procedure used during training. Currently random destroy (R), tour-based destroy (T) and point-based destroy (P) are implemented.

To allow split deliveries during training use the `--allow_split_delivery` flag.

For each run, a separate folder is created in the *runs* directory. The associated log file can be used to monitor the progress of the training. The model weights of the model with the best validation performance is stored in the subdirectory *models*. A description of the destroy operator (i.e., destroy procedure + degree of destruction) used during the training is stored in the model data and is used when the model is employed in a search. 

The definition of the instance group properties can be found in the folder *dataset_blueprints* (e.g., in fhe file *XE.py* for the XE_instance group). If you want to solve instances with different properties (e.g., number of customers) you need to define a corresponding new instance blueprint in the code.


### Instance Generator (XE instances)

You can generate new instances from the XE instance groups and save them as .vrp files that are  supported by other (heuristic) solvers (e.g. LKH3). For example, the following command generates 10 instances of the XE_1 instance group (using 0 as a seed) and saves them in *data/XE_1_instances*.

```
python3 generate_instances.py --data_dir data/XE_1_instances --instance_blueprint XE_1 --dataset_size 10 --seed 0
```

## Acknowledgements

Parts of the code are based on https://github.com/mveres01/pytorch-drl4vrp which is a great starting point for learning/implementing deep reinforcement learning approaches for vehicle routing problems. Furthermore, some parts of the code are inspired by https://github.com/wouterkool/attention-learn-to-route and https://github.com/OptMLGroup/VRP-RL. 