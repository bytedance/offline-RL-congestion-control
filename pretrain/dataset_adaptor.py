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


def generate_episode(call_data):
    observations = np.asarray(call_data['observations'], dtype=np.float32)
    bandwidth_predictions = np.expand_dims(np.asarray(call_data['bandwidth_predictions'], dtype=np.float32),
                                            axis=1)

    terminals = np.zeros_like(bandwidth_predictions)
    terminals[-1][-1] = 1

    rewards = np.ones_like(bandwidth_predictions)

    next_observation = np.zeros_like(observations)
    next_observation[:-1, :] = observations[1:, :]

    episode = collections.deque(
        np.concatenate((observations, bandwidth_predictions, rewards, next_observation, terminals), axis=1))
    return episode
     

class PretrainDatasetAdaptor():
    def __init__(self, state_dim, buffer_size, data_files):
        self.state_dim = state_dim
        self.buffer_size = buffer_size
        self.dataset_postfix = ".json"

        self.buffer = ReplayBuffer(buffer_size)
        self.data_files = data_files

    def adapt_all_data(self):
        for file_name in tqdm(self.data_files, desc="Processing", position=0):
            with open(file_name, 'rb') as file:
                    call_data = json.load(file)

            if call_data is None:
                continue

            episode = generate_episode(call_data)

            self.buffer.add_episode(episode)

    def generate_datasets(self):
        datafile_to_process = random.sample(self.data_files, len(self.data_files))
        transition_length = 0
        for file_name in tqdm(datafile_to_process, desc="Processing", position=0):
            with open(file_name, 'rb') as file:
                    call_data = json.load(file)

            if call_data is None:
                continue

            episode = generate_episode(call_data)
            self.buffer.add_episode(episode)

            transition_length += len(episode)
            if transition_length >= self.buffer_size:
                return

    def sample_state_and_label(self, num):
        b_s, b_a, b_r, b_ns, b_d = self.buffer.np_array_sample(self.state_dim, num)
        return b_s, b_ns

    def get_all_state_and_label(self):
        b_s, b_ns = self.buffer.get_pretrain_state_and_label(self.state_dim)
        return b_s, b_ns