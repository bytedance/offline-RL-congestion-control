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
import glob
import json
import os
import numpy as np
import collections
import torch
from tqdm import tqdm
import onnxruntime as ort
import matplotlib
import matplotlib.pyplot as plt
from utils import ReplayBuffer
from utils import distance_info
import pickle

matplotlib.use('Agg')

eval_data_directory_name = 'eval_dataset'
np.set_printoptions(formatter={'float': '{:.2f}'.format})

def generate_episode(call_data, reward_func):
    observations = np.asarray(call_data['observations'], dtype=np.float32)
    bandwidth_predictions = np.expand_dims(np.asarray(call_data['bandwidth_predictions'], dtype=np.float32),axis=1)

    true_capacity = np.expand_dims(np.asarray(call_data['true_capacity'], dtype=np.float32), axis=1)

    terminals = np.zeros_like(bandwidth_predictions)
    terminals[-1][-1] = 1

    rewards = reward_func(bandwidth_predictions, true_capacity)

    next_observation = np.zeros_like(observations)
    next_observation[:-1, :] = observations[1:, :]

    episode = collections.deque(
        np.concatenate((observations, bandwidth_predictions, rewards, next_observation, terminals), axis=1))
    
    return episode, true_capacity


class OfflineEvaluator():
    def __init__(self, task_dir, test_data_dir, validation_data_dir, state_dim, device, reward_func, eval_num_limit=200):
        self.test_data_dir = test_data_dir
        self.eval_num_limit = eval_num_limit
        self.reward_func = reward_func
        self.task_dir = task_dir
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        self.call_datas = []
        self.test_data_files = []
        self.validation_data_files = []

        test_data_files = glob.glob(os.path.join(test_data_dir, f'*.json'), recursive=True)
        if len(test_data_files) == 0:
            raise Exception("Test data directory is empty. Please run test_set_generator.py first")
        self.test_data_files = test_data_files

        validation_data_files = glob.glob(os.path.join(validation_data_dir, f'*.json'), recursive=True)
        if len(validation_data_files) == 0:
            raise Exception("Validation data directory is empty. Please run test_set_generator.py first")
        self.validation_data_files = validation_data_files

        self.best_reward_step = 0
        self.best_dis_step = 0
        self.max_reward = 0
        self.min_dis = 0
        self.smooth_reward = 0
        self.eval_reward_list = []
        self.eval_reward_list_step = []
        self.train_reward_list = []
        self.train_reward_list_step = []

        self.eval_dis_info = {}
        self.eval_dis_ratio_info = {}
        self.train_dis_info = {}
        self.train_dis_ratio_info = {}

        self.state_dim = state_dim
        self.device = device
        self.evaluate_state = None
        self.eval_label = None
        self.train_state = None
        self.train_label = None
        self.train_qos = None
        self.baseline_dict = {}
        self.max_ratio_step = 0

        self.init_eval_state()
        self.init_train_state()
        self.init_baseline_data()

        self.baseline_ratio_info = {
            "0_50": 0,
            "50_80": 0,
            "80_110": 0,
            "110_150": 0,
            "150_200": 0,
            "200_300": 0,
            "300_500": 0,
            "500": 0
        }
        self.baseline_dis_mean = []
        self.baseline_ratio_has_record = False

    def init_eval_state(self):
        evaluate_buffer = ReplayBuffer(450000)
        self.test_data_files = self.test_data_files[:self.eval_num_limit]
        for filename in tqdm(self.test_data_files, desc="Loading Evaluate Dataset"):
            call_data = None
            with open(filename, 'rb') as file:
                call_data = json.load(file)
                self.call_datas.append(call_data)

            if call_data is None:
                continue
            
            episode, true_capacity = generate_episode(call_data, self.reward_func)
            
            evaluate_buffer.add_episode(episode)
            evaluate_buffer.add_y_label(true_capacity)

        state, label = evaluate_buffer.get_state_and_label(self.state_dim)
        self.evaluate_state = torch.tensor(state, dtype=torch.float).to(self.device)
        self.eval_label = label

    def init_baseline_data(self):
        with open(os.path.join(self.test_data_dir, 'baseline_eval.pkl'), 'rb') as file:
            self.baseline_dict = pickle.load(file)

    def init_train_state(self):
        train_eval_buffer = ReplayBuffer(450000)
        self.validation_data_files = self.validation_data_files[:self.eval_num_limit]
        for filename in tqdm(self.validation_data_files, desc="Loading train dataset for evaluating"):
            call_data = None
            with open(filename, 'rb') as file:
                call_data = json.load(file)

            if call_data is None:
                continue

            episode, true_capacity = generate_episode(call_data, self.reward_func)

            train_eval_buffer.add_episode(episode)
            train_eval_buffer.add_y_label(true_capacity)

        state, label = train_eval_buffer.get_state_and_label(self.state_dim)
        self.train_state = torch.tensor(state, dtype=torch.float).to(self.device)
        self.train_label = label

    def evaluate(self, agent, step):
        dummy_hidden_state = torch.rand(1, 1).to(self.device)
        dummy_cell_state = torch.rand(1, 1).to(self.device)
        eval_action, _, _ = agent.take_best_action(self.evaluate_state, dummy_hidden_state, dummy_cell_state)
        eval_action = eval_action.detach().cpu().numpy()
        eval_action = eval_action[:, :, 0]
        eval_rewards = self.reward_func(bandwidth_predictions=eval_action,
                                        true_capacity=self.eval_label)
        self.eval_reward_list_step.append(step)
        self.eval_reward_list.append(eval_rewards.mean())

        # process distance
        dis, ratio_info = distance_info(eval_action, self.eval_label)
        self.eval_dis_info[str(step)] = dis.mean()
        self.eval_dis_ratio_info[str(step)] = ratio_info

        train_action, _, _ = agent.take_best_action(self.train_state, dummy_hidden_state, dummy_cell_state)
        train_action = train_action.detach().cpu().numpy()
        train_action = train_action[:, :, 0]
        train_rewards = self.reward_func(bandwidth_predictions=train_action,
                                         true_capacity=self.train_label)
        self.train_reward_list_step.append(step)
        self.train_reward_list.append(train_rewards.mean())

        # process distance
        dis, ratio_info = distance_info(train_action, self.train_label)
        self.train_dis_info[str(step)] = dis.mean()
        self.train_dis_ratio_info[str(step)] = ratio_info

        # mark_best_model
        if self.best_reward_step == 0 or self.eval_reward_list[-1] > self.max_reward:
            self.best_reward_step = step
            self.max_reward = self.eval_reward_list[-1]

        if self.best_dis_step == 0 or self.eval_dis_info[str(step)] < self.min_dis:
            self.best_dis_step = step
            self.min_dis = self.eval_dis_info[str(step)]

    def draw_best_model_figs(self, step, directory_postfix):
        onnx_model_path = os.path.join(self.task_dir, "model_" + str(step) + '.onnx')
        ort_session = ort.InferenceSession(onnx_model_path)
        figs_path = os.path.join(self.task_dir, "model_" + str(step) + '_' + directory_postfix)
        if not os.path.exists(figs_path):
            os.mkdir(figs_path)

        episode_len = len(self.call_datas)

        for i, call_data in enumerate(tqdm(self.call_datas, desc="Best Model Evaluating")):
            observations = np.asarray(call_data['observations'], dtype=np.float32)
            bandwidth_predictions = np.expand_dims(np.asarray(call_data['bandwidth_predictions'], dtype=np.float32),
                                                   axis=1)
            true_capacity = np.expand_dims(np.asarray(call_data['true_capacity'], dtype=np.float32), axis=1)
            terminals = np.zeros_like(bandwidth_predictions)
            terminals[-1] = 1

            model_predictions = []

            hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

            for t in range(observations.shape[0]):
                feed_dict = {'obs': observations[t:t + 1, :].reshape(1, 1, -1), 'hidden_states': hidden_state,
                             'cell_states': cell_state}
                bw_prediction, hidden_state, cell_state = ort_session.run(None, feed_dict)
                model_predictions.append(bw_prediction[0][0][0])
            model_predictions = np.expand_dims(np.asarray(model_predictions, dtype=np.float32), axis=1)

            fig = plt.figure(figsize=(8, 8))
            time_s = np.arange(0, observations.shape[0] * 60, 60) / 1000
            baseline_bw = []
            if len(self.baseline_dict) != 0:
                baseline_bw = self.baseline_dict[os.path.basename(self.test_data_files[i])]
            plt.subplot(2, 1, 1)
            plt.plot(time_s, true_capacity / 1000, label='True Capacity', color='k')
            plt.plot(time_s, bandwidth_predictions / 1000, label='BW Estimator ' + call_data['policy_id'], color='r')
            if len(baseline_bw) != 0:
                plt.plot(time_s, baseline_bw, label='MMsys Baseline', color='b')
            plt.plot(time_s, model_predictions / 1000, label="model_" + str(step), color='g')
            max_capa = true_capacity[~np.isnan(true_capacity)].max() / 1000
            plt.ylim([0, 2 * max_capa])
            plt.ylabel("Bandwidth (Kbps)")
            plt.xlabel("Call Duration (second)")
            plt.grid(True)
            plt.legend()

            # record ratio
            if not self.baseline_ratio_has_record:
                dis, ratio_dict = distance_info(np.expand_dims(np.asarray(baseline_bw * 1000, dtype=np.float32), axis=1), true_capacity)
                for k, v in self.baseline_ratio_info.items():
                    self.baseline_ratio_info[k] += ratio_dict[k] / episode_len

                self.baseline_dis_mean.append(dis.mean())

            baseline_reward = self.reward_func(
                bandwidth_predictions=np.expand_dims(np.asarray(baseline_bw, dtype=np.float32), axis=1),
                true_capacity=true_capacity)

            plt.subplot(2, 1, 2)
            rewards = self.reward_func(bandwidth_predictions=model_predictions,
                                       true_capacity=true_capacity,return_separate_item=True)
            total_rewards = np.zeros_like(bandwidth_predictions)
            for k, v in rewards.items():
                plt.plot(time_s, v, label=k)
                total_rewards += v
            plt.plot(time_s, total_rewards, label="total reward")
            plt.ylabel("Reward")
            plt.xlabel("Offline reward")
            plt.grid(True)
            plt.legend()

            plt.savefig(os.path.join(figs_path, os.path.basename(self.test_data_files[i]).replace(".json", ".png")))
            plt.savefig(os.path.join(figs_path, os.path.basename(self.test_data_files[i]).replace(".json", ".pdf")))
            plt.close()

        self.baseline_ratio_has_record = True

    def end_process(self, draw_max_ratio=False):
        self.draw_best_model_figs(self.best_reward_step, "best_reward_figs")
        self.draw_list()
        self.record_dis_info()
        if draw_max_ratio:
            self.draw_best_model_figs(self.max_ratio_step, "maximum_ratio_figs")

    def draw_list(self):
        plt.plot(self.eval_reward_list_step, self.eval_reward_list)
        plt.xlabel("reward")
        plt.savefig(os.path.join(self.task_dir, "eval_reward_list.png"))
        plt.savefig(os.path.join(self.task_dir, "eval_reward_list.pdf"))
        plt.close()

        plt.plot(self.train_reward_list_step, self.train_reward_list)
        plt.xlabel("reward")
        plt.savefig(os.path.join(self.task_dir, "train_reward_list.png"))
        plt.savefig(os.path.join(self.task_dir, "train_reward_list.pdf"))
        plt.close()

        ratio_0_80 = []
        ratio_80_110 = []
        ratio_110_200 = []
        ratio_200_above = []
        maximum_ratio = 0
        maximum_step = ""
        for k, v in self.eval_dis_ratio_info.items():
            ratio_0_80.append(v["0_50"] + v["50_80"])
            ratio_80_110.append(v["80_110"])
            ratio_110_200.append(v["110_150"] + v["150_200"])
            ratio_200_above.append(v["200_300"] + v["300_500"] + v["500"])
            if v["80_110"] > maximum_ratio:
                maximum_ratio = v["80_110"]
                maximum_step = k

        plt.plot(self.eval_reward_list_step, ratio_0_80, label="0%~80%")
        plt.plot(self.eval_reward_list_step, ratio_80_110, label="80%~110%")
        plt.plot(self.eval_reward_list_step, ratio_110_200, label="110%~200%")
        plt.plot(self.eval_reward_list_step, ratio_200_above, label="200% above")
        plt.xlabel("Ratio info, max ratio is " + str(maximum_ratio) + ", step is "+ maximum_step)
        plt.legend()
        plt.savefig(os.path.join(self.task_dir, "ratio_info.png"))
        plt.savefig(os.path.join(self.task_dir, "ratio_info.pdf"))
        plt.close()

    def record_dis_info(self):
        maximum_step = 0
        maximum_80_110_ratio = 0
        minimum_ratio_info = {}
        with open(os.path.join(self.task_dir, "dis_info.txt"), 'w+') as f:
            f.write('eval_data \n')
            for k, v in self.eval_dis_info.items():
                f.write(k + ": mean is " + str(v) + ". Ratio_info:" + str(self.eval_dis_ratio_info[k]))
                f.write('\n')
                if (maximum_step == 0 and maximum_80_110_ratio == 0) or self.eval_dis_ratio_info[k]["80_110"] > maximum_80_110_ratio:
                    maximum_step = k
                    maximum_80_110_ratio = self.eval_dis_ratio_info[k]["80_110"]
                    minimum_ratio_info = self.eval_dis_ratio_info[k]

            f.write('\ntrain_data \n')
            for k, v in self.train_dis_info.items():
                f.write(k + ": mean is " + str(v) + ". Ratio_info:" + str(self.train_dis_ratio_info[k]))
                f.write('\n')

            self.max_ratio_step = maximum_step

            f.write("\nThe maximum ratio step on eval data is:" + maximum_step + ", the 80_110 ratio is " + str(
                maximum_80_110_ratio) + ".\nRatio_info:" + str(minimum_ratio_info))
            f.write('\n\n\n')

            for k, v in self.baseline_ratio_info.items():
                self.baseline_ratio_info[k] = round(self.baseline_ratio_info[k], 4)

            if self.baseline_ratio_has_record:
                f.write("MMsys baseline dis_mean:" + str(np.array(self.baseline_dis_mean).mean()) + "\n")
                f.write("MMsys baseline Ratio_info:" + str(self.baseline_ratio_info))
