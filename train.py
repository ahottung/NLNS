import numpy as np
import torch
import torch.optim as optim
import os
from search import destroy_instances
from copy import deepcopy
import logging
import datetime
from search_batch import lns_batch_search
import repair
import main
from vrp.data_utils import create_dataset
from search import LnsOperatorPair


def train_nlns(actor, critic, run_id, config):
    batch_size = config.batch_size

    logging.info("Generating training data...")
    # Create training and validation set. The initial solutions are created greedily
    training_set = create_dataset(size=batch_size * config.nb_batches_training_set, config=config,
                                  create_solution=True, use_cost_memory=False)
    logging.info("Generating validation data...")
    validation_instances = create_dataset(size=config.valid_size, config=config, seed=config.validation_seed,
                                          create_solution=True)

    actor_optim = optim.Adam(actor.parameters(), lr=config.actor_lr)
    actor.train()
    critic_optim = optim.Adam(critic.parameters(), lr=config.critic_lr)
    critic.train()

    losses_actor, rewards, diversity_values, losses_critic = [], [], [], []
    incumbent_costs = np.inf
    start_time = datetime.datetime.now()

    logging.info("Starting training...")
    for batch_idx in range(1, config.nb_train_batches + 1):
        # Get a batch of training instances from the training set. Training instances are generated in advance, because
        # generating them is expensive.
        training_set_batch_idx = batch_idx % config.nb_batches_training_set
        tr_instances = [deepcopy(instance) for instance in
                        training_set[training_set_batch_idx * batch_size: (training_set_batch_idx + 1) * batch_size]]

        # Destroy and repair the set of instances
        destroy_instances(tr_instances, config.lns_destruction, config.lns_destruction_p)
        costs_destroyed = [instance.get_costs_incomplete(config.round_distances) for instance in tr_instances]
        tour_indices, tour_logp, critic_est = repair.repair(tr_instances, actor, config, critic)
        costs_repaired = [instance.get_costs(config.round_distances) for instance in tr_instances]

        # Reward/Advantage computation
        reward = np.array(costs_repaired) - np.array(costs_destroyed)
        reward = torch.from_numpy(reward).float().to(config.device)
        advantage = reward - critic_est

        # Actor loss computation and backpropagation
        actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
        actor_optim.step()

        # Critic loss computation and backpropagation
        critic_loss = torch.mean(advantage ** 2)
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
        critic_optim.step()

        rewards.append(torch.mean(reward.detach()).item())
        losses_actor.append(torch.mean(actor_loss.detach()).item())
        losses_critic.append(torch.mean(critic_loss.detach()).item())

        # Replace the solution of the training set instances with the new created solutions
        for i in range(batch_size):
            training_set[training_set_batch_idx * batch_size + i] = tr_instances[i]

        # Log performance every 250 batches
        if batch_idx % 250 == 0 and batch_idx > 0:
            mean_loss = np.mean(losses_actor[-250:])
            mean_critic_loss = np.mean(losses_critic[-250:])
            mean_reward = np.mean(rewards[-250:])
            logging.info(
                f'Batch {batch_idx}/{config.nb_train_batches}, repair costs (reward): {mean_reward:2.3f}, loss: {mean_loss:2.6f}'
                f', critic_loss: {mean_critic_loss:2.6f}')

        # Evaluate and save model every 5000 batches
        if batch_idx % 5000 == 0 or batch_idx == config.nb_train_batches:
            mean_costs = lns_validation_search(validation_instances, actor, config)
            model_data = {
                'parameters': actor.state_dict(),
                'model_name': "VrpActorModel",
                'destroy_operation': config.lns_destruction,
                'p_destruction': config.lns_destruction_p,
                'code_version': main.VERSION
            }

            if config.split_delivery:
                problem_type = "SD"
            else:
                problem_type = "C"
            torch.save(model_data, os.path.join(config.output_path, "models",
                                                "model_{0}_{1}_{2}_{3}_{4}.pt".format(problem_type,
                                                                                      config.instance_blueprint,
                                                                                      config.lns_destruction,
                                                                                      config.lns_destruction_p,
                                                                                      run_id)))
            if mean_costs < incumbent_costs:
                incumbent_costs = mean_costs
                incumbent_model_path = os.path.join(config.output_path, "models",
                                                    "model_incumbent_{0}_{1}_{2}_{3}_{4}.pt".format(problem_type,
                                                                                                    config.instance_blueprint,
                                                                                                    config.lns_destruction,
                                                                                                    config.lns_destruction_p,
                                                                                                    run_id))
                torch.save(model_data, incumbent_model_path)

            runtime = (datetime.datetime.now() - start_time)
            logging.info(
                f"Validation (Batch {batch_idx}) Costs: {mean_costs:.3f} ({incumbent_costs:.3f}) Runtime: {runtime}")
    return incumbent_model_path


def lns_validation_search(validation_instances, actor, config):
    validation_instances_copies = [deepcopy(instance) for instance in validation_instances]
    actor.eval()
    operation = LnsOperatorPair(actor, config.lns_destruction, config.lns_destruction_p)
    costs, _ = lns_batch_search(validation_instances_copies, config.lns_max_iterations,
                                config.lns_timelimit_validation, [operation], config)
    actor.train()
    return np.mean(costs)
