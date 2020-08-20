import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser(description='Neural Large Neighborhood Search')

    parser.add_argument('--mode', default='train', type=str, choices=['train', 'eval_single', 'eval_batch'])
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--validation_seed', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--round_distances', default=False, action='store_true')
    parser.add_argument('--allow_split_delivery', dest='split_delivery', default=False, action='store_true')

    # Training
    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pointer_hidden_size', default=128, type=int)
    parser.add_argument('--critic_hidden_size', default=128, type=int)
    parser.add_argument('--nb_train_batches', default=250000, type=int)
    parser.add_argument('--nb_batches_training_set', default=1500, type=int)
    parser.add_argument('--lns_destruction_p', default=0.3, type=float)
    parser.add_argument('--lns_destruction', default="P", type=str)
    parser.add_argument('--instance_blueprint', default="ALTR_20", type=str)
    parser.add_argument('--valid_size', default=500, type=int)

    # Search
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--instance_path', default=None, type=str)
    parser.add_argument('--test_size', default=2000, type=int)
    parser.add_argument('--lns_nb_cpus', default=1, type=int)
    parser.add_argument('--lns_timelimit', default=180, type=int)
    parser.add_argument('--lns_max_iterations', default=50000, type=int)  # Is ignored by single instance search
    parser.add_argument('--lns_batch_size', default=300, type=int)
    parser.add_argument('--lns_t_max', default=1000, type=int)
    parser.add_argument('--lns_t_min', default=10, type=float)
    parser.add_argument('--lns_reheating_nb', default=5, type=int)
    parser.add_argument('--lns_Z_param', default=0.8, type=float)
    parser.add_argument('--lns_adaptive_search', default=False, action='store_true')
    parser.add_argument('--nb_runs', default=1, type=int)

    config = parser.parse_args()

    config_d = vars(config)
    config_d['device'] = torch.device(config.device)
    config_d['lns_timelimit_validation'] = config.lns_timelimit * (config.valid_size / config.test_size)

    return config


