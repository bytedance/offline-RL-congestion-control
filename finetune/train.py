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

import json
import os
from utils import *
import random
import glob
from iql_agent import IQL
from tqdm import tqdm
from datetime import datetime
from dataset_adaptor import DatasetAdaptor
from offline_evaluator import OfflineEvaluator
import sys
sys.path.append("../pretrain")

def set_random_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_num)

def generate_rate_limit(current_step):
    factor = 100

    if 0 <= current_step % 1000000 < 1000 * factor:
        low_bitrate_limit = 50000
        high_birate_limit = 100000
    elif 1000 * factor <= current_step % 1000000 < 2000 * factor:
        low_bitrate_limit = 50000
        high_birate_limit = 200000
    elif 2000 * factor <= current_step % 1000000 < 3000 * factor:
        low_bitrate_limit = 50000
        high_birate_limit = 300000
    elif 4000 * factor <= current_step % 1000000 < 5000 * factor:
        low_bitrate_limit = 50000
        high_birate_limit = 500000
    elif 5000 * factor <= current_step % 1000000 < 6000 * factor:
        low_bitrate_limit = 50000
        high_birate_limit = 1000000
    elif 6000 * factor <= current_step % 1000000 < 7000 * factor:
        low_bitrate_limit = 50000
        high_birate_limit = 5000000
    else:
        low_bitrate_limit = 50000
        high_birate_limit = 10000000

    return low_bitrate_limit, high_birate_limit


if __name__ == "__main__":
    with open("./finetune_config.json", 'r') as f:
        train_config = json.load(f)

    task_name = train_config["task_name"]
    task_name = datetime.now().strftime("%Y%m%d_%H%M") + "_" + task_name

    buffer_size = train_config["buffer_size"]
    batch_size = train_config["batch_size"]
    max_total_step = train_config["total_step"]
    evaluate_and_model_save_interval = train_config["evaluate_and_model_save_interval"]
    buffer_update_interval = train_config["buffer_update_interval"]
    final_evaluate_step_interval = train_config["final_large_evaluate_step_interval"]

    seed = 666
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(seed)
    state_dim = 150

    # path information
    eval_result_dir = train_config["path"]["result_path"]
    emulated_dataset_dir = train_config["path"]["emulated_dataset_path"]
    val_dataset_dir = train_config["path"]["validation_set_path"]
    test_dataset_dir = train_config["path"]["test_set_path"]

    # finetune setting
    finetune_setting = train_config["fine_tuned_setting"]
    is_fine_tuned = finetune_setting["enable_fine_tuned"]
    finetune_model_path = finetune_setting["pretrain_model_path"]

    agent = IQL(device=device, is_fine_tuned=is_fine_tuned)
    
    if is_fine_tuned:
        pretrain_model = torch.load(finetune_model_path)
        agent.load_pretrain_encoder(pretrain_model.agent.encoder)
        print("loaded fine tuned success!")

    if not os.path.exists(eval_result_dir):
        os.makedirs(eval_result_dir)

    task_dir = os.path.join(eval_result_dir, task_name)
    eval_num_limit = train_config["eval_num_limit"]
    offline_evaluator = OfflineEvaluator(task_dir,
                                         test_dataset_dir,
                                         val_dataset_dir,
                                         state_dim,
                                         device,
                                         reward_func,
                                         eval_num_limit)
    
    # load data
    dataset_postfix = ".json"
    emulated_data_files = glob.glob(os.path.join(emulated_dataset_dir, f'*' + dataset_postfix), recursive=True)
    dataset_adaptor = DatasetAdaptor(buffer_size, reward_func, emulated_data_files)

    low_bitrate_limit = 50000
    high_bitrate_limit = 300000

    dataset_adaptor.generate_datasets(low_bitrate_target=low_bitrate_limit,
                                      high_bitrate_target=high_bitrate_limit)
    replay_buffer = dataset_adaptor.get_dataset()

    observation_scaler = MinMaxScalers(device)
    observation_scaler.fit_with_data(obs_min.to(device), obs_max.to(device))
    action_scaler = MinMaxScalers(device)
    action_scaler.fit_with_data(bwe_min.to(device), bwe_max.to(device))
    agent.update_scaler(observation_scaler, action_scaler)

    print("Start training:" + datetime.now().strftime("Date is %Y%m%d, time is %H:%M"))
    start_training_time = datetime.now()

    current_loop = 0
    current_step = 0
    
    n_loop = max_total_step // buffer_update_interval

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
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.np_array_sample(state_dim, batch_size)
            hidden_state, cell_state = np.zeros((batch_size, 1), dtype=np.float32), np.zeros((batch_size, 1),
                                                                                             dtype=np.float32)

            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                               'dones': b_d, 'hidden_states': hidden_state, 'cell_states': cell_state}
            agent.train(transition_dict)
            current_step += 1

            if current_step % evaluate_and_model_save_interval == 0:
                agent.freeze_actor()
                offline_evaluator.evaluate(agent, current_step)
                agent.unfreeze_actor()
                model_name = "model_" + str(current_step)
                model_dir = os.path.join(task_dir, model_name + ".onnx")
                agent.save_policy(model_dir)
                pt_dir = os.path.join(task_dir, model_name + ".pt")
                torch.save(agent, pt_dir)

        print('------------------------- epoch:' + str(current_loop) + ' -------------------------')

        duration = datetime.now() - now
        print("duration:" + str(duration) + " second_per_step:" + str(duration.seconds / buffer_update_interval))

        if current_loop == n_loop:
            break

        low_bitrate_limit, high_bitrate_limit = generate_rate_limit(current_step)
        dataset_adaptor.generate_datasets(low_bitrate_target=low_bitrate_limit,
                                          high_bitrate_target=high_bitrate_limit)
        replay_buffer = dataset_adaptor.get_dataset()

        if current_step % final_evaluate_step_interval == 0:
            offline_evaluator.end_process()

    offline_evaluator.end_process(True)
    print("Train done, time cost is " + str(datetime.now() - start_training_time))
    print(datetime.now().strftime("Date is %Y%m%d, time is %H:%M"))

