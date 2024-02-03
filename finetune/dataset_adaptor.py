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
import random
import json
import numpy as np
import collections
from utils import ReplayBuffer
from tqdm import tqdm


class DatasetAdaptor():
    def __init__(self, buffer_size, reward_func, data_files):
        self.buffer_size = buffer_size
        self.data_files = data_files
        self.buffer = ReplayBuffer(buffer_size)
        self.datafile_to_process = self.data_files
        self.reward_func = reward_func

    def generate_datasets(self, low_bitrate_target=0, high_bitrate_target=0):
        self.datafile_to_process = random.sample(self.data_files, len(self.data_files))
        transition_length = 0
        for file_name in tqdm(self.datafile_to_process, desc="Processing", position=0):
            call_data = None
            with open(file_name, 'rb') as file:
                    call_data = json.load(file)

            if call_data is None:
                continue

            observations = np.asarray(call_data['observations'], dtype=np.float32)
            bandwidth_predictions = np.expand_dims(np.asarray(call_data['bandwidth_predictions'], dtype=np.float32),
                                                   axis=1)

            true_capacity = 0
            if 'true_capacity' in call_data:
                true_capacity = np.expand_dims(np.asarray(call_data['true_capacity'], dtype=np.float32), axis=1)

                if low_bitrate_target != 0 and high_bitrate_target != 0:
                    if low_bitrate_target > true_capacity[~np.isnan(true_capacity)].mean() or high_bitrate_target < true_capacity[~np.isnan(true_capacity)].mean():
                        continue

            terminals = np.zeros_like(bandwidth_predictions)
            terminals[-1][-1] = 1

            rewards = self.reward_func(bandwidth_predictions, true_capacity)

            next_observation = np.zeros_like(observations)
            next_observation[:-1, :] = observations[1:, :]

            # use np.tile to expand (batch_size, 1) to (batch_size, 150)
            # after index filter, observation become (batch_size * 150, ) need to be reshaped
            filtered_obs = observations[np.tile(~np.isnan(true_capacity), 150)].reshape(-1, 150)
            filtered_bwe = bandwidth_predictions[~np.isnan(true_capacity)].reshape(-1, 1)
            filtered_rwd = rewards[~np.isnan(true_capacity)].reshape(-1, 1)
            filtered_nobs = next_observation[np.tile(~np.isnan(true_capacity), 150)].reshape(-1, 150)
            filtered_tml = terminals[~np.isnan(true_capacity)].reshape(-1, 1)
            filtered_cap = true_capacity[~np.isnan(true_capacity)].reshape(-1, 1)

            b_c_ratio = filtered_bwe / filtered_cap
            bool_index = np.where(np.logical_and(0.8 <= b_c_ratio, b_c_ratio <= 1.1))
            filtered_obs = filtered_obs[bool_index[0], :]
            filtered_bwe = filtered_bwe[bool_index[0]]
            filtered_rwd = filtered_rwd[bool_index[0]]
            filtered_nobs = filtered_nobs[bool_index[0], :]
            filtered_tml = filtered_tml[bool_index[0]]

            episode = collections.deque(np.concatenate((filtered_obs, filtered_bwe, filtered_rwd, filtered_nobs, filtered_tml), axis=1))
            self.buffer.add_episode(episode)

            transition_length += len(episode)
            if transition_length >= self.buffer_size:
                return

    def get_dataset(self):
        return self.buffer





