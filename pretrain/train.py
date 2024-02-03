# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import random
import numpy as np
from datetime import datetime
import glob
from tqdm import tqdm
import torch
from utils import MinMaxScalers, obs_min, obs_max
from dataset_adaptor import PretrainDatasetAdaptor
from pretrain_agent import PretrainAgent
import matplotlib.pyplot as plt


def set_random_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_num)


def plot(loss, dir):
    # draw figures
    plt.figure(figsize=(12, 6))
    plt.title(f"loss")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot(np.arange(len(loss)), loss, 'r', linewidth=2.0)
    plt.savefig(os.path.join(dir, "loss.pdf"), dpi=200)
    plt.savefig(os.path.join(dir, "loss.png"), dpi=200)
    plt.close()


if __name__ == '__main__':
    with open("pretrain_config.json", 'r') as f:
        train_config = json.load(f)

    buffer_size = train_config["buffer_size"]
    batch_size = train_config["batch_size"]
    max_total_step = train_config["total_step"]
    evaluate_and_model_save_interval = train_config["evaluate_and_model_save_interval"]
    buffer_update_interval = train_config["buffer_update_interval"]
    plot_interval = train_config["plot_interval"]

    # path information
    eval_result_dir = train_config["path"]["result_path"]
    pre_train_emulated_dataset_dir = train_config["path"]["emulated_dataset_path"]
    pre_train_testbed_dataset_dir = train_config["path"]["testbed_dataset_path"]
    val_dataset_dir = train_config["path"]["validation_set_path"]
    test_dataset_dir = train_config["path"]["test_set_path"]

    state_dim = 150
    current_loop = 0
    current_step = 0

    # set random seed
    set_random_seed(666)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(eval_result_dir):
        os.makedirs(eval_result_dir)

    task_name = "pretrain_" + datetime.now().strftime("%Y%m%d_%H%M")
    task_dir = os.path.join(eval_result_dir, task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    # load data
    dataset_postfix = ".json"
    emulated_data_files = glob.glob(os.path.join(pre_train_emulated_dataset_dir, f'*' + dataset_postfix), recursive=True)
    testbed_data_files = glob.glob(os.path.join(pre_train_testbed_dataset_dir, f'*' + dataset_postfix), recursive=True)
    train_data_files = emulated_data_files + testbed_data_files
    train_dataloader = PretrainDatasetAdaptor(state_dim, buffer_size, train_data_files)
    train_dataloader.generate_datasets()

    val_data_files = glob.glob(os.path.join(val_dataset_dir, f'*' + dataset_postfix), recursive=True)
    val_dataloader = PretrainDatasetAdaptor(state_dim, 450000, val_data_files)
    val_dataloader.adapt_all_data()

    test_data_files = glob.glob(os.path.join(test_dataset_dir, f'*' + dataset_postfix), recursive=True)
    test_dataloader = PretrainDatasetAdaptor(state_dim, 450000, test_data_files)
    test_dataloader.adapt_all_data()

    # agent
    agent = PretrainAgent()
    observation_scaler = MinMaxScalers(device)
    observation_scaler.fit_with_data(obs_min, obs_max)
    agent.set_scalar(observation_scaler)

    # training process
    n_loop = max_total_step // buffer_update_interval

    val_loss_list = []
    test_loss_list = []

    min_loss = 100000000
    model_name = "model"

    val_input, val_label = val_dataloader.get_all_state_and_label()
    test_input, test_label = test_dataloader.get_all_state_and_label()

    for epoch in range(1, n_loop + 1):
        current_loop += 1
        range_gen = tqdm(
            range(buffer_update_interval),
            disable=False,
            desc=f"Epoch {int(epoch)}/{n_loop}",
            position=0
        )

        now = datetime.now()

        for itr in range_gen:
            train_data, label = train_dataloader.sample_state_and_label(batch_size)
            agent.train(train_data, label)
            current_step += 1

            if current_step % evaluate_and_model_save_interval == 0:
                model_name = "model_" + str(current_step)
                pt_dir = os.path.join(task_dir, model_name + ".pt")
                torch.save(agent, pt_dir)

            if current_step % plot_interval == 0:
                plot(test_loss_list, task_dir)

        val_loss = agent.evaluate(val_input, val_label)
        val_loss_list.append(val_loss)

        test_loss = agent.evaluate(test_input, test_label)
        test_loss_list.append(test_loss)

        if test_loss < min_loss:
            min_loss = test_loss
            best_model_name = model_name

        print('------------------------- epoch:' + str(current_loop) + ' -------------------------')

        if current_loop <= n_loop:
            train_dataloader.generate_datasets()
        duration = datetime.now() - now
        print("duration:" + str(duration) + " second_per_step:" + str(duration.seconds / buffer_update_interval) + " test loss:" + str(test_loss), " best model:" + best_model_name)




