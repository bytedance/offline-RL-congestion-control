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

import numpy as np
import torch
import collections
import random


def distance_info(action, label):
    dis = action - label

    dis = dis[~np.isnan(dis)]

    # process exceed
    label[label < 5000] = 5000
    ratio = action / label
    ratio = ratio[~np.isnan(ratio)]
    ratio[np.where(ratio > 1)] = ratio[np.where(ratio > 1)] * 5
    ceil_ratio = np.ceil(ratio)
    ceil_ratio = ceil_ratio / 5
    ceil_ratio_power = ceil_ratio * ceil_ratio

    # process inverse
    inver_ratio = label / action
    inver_ratio = inver_ratio[~np.isnan(inver_ratio)]
    inver_ratio[np.where(inver_ratio > 1)] = inver_ratio[np.where(inver_ratio > 1)] * 5
    ceil_inverse_ratio = np.ceil(inver_ratio)
    ceil_inverse_ratio = ceil_inverse_ratio / 5
    factor = ceil_ratio_power * ceil_inverse_ratio

    dis = dis / 1000
    dis = dis * factor
    dis = abs(dis)

    total_num = len(ceil_ratio)

    actual_ratio = action / label
    actual_ratio = actual_ratio[~np.isnan(actual_ratio)]

    ratio_dict = {
        "0_50": round(len(actual_ratio[np.where(np.logical_and(0 < actual_ratio, actual_ratio <= 0.5))]) / total_num, 4),
        "50_80": round(len(actual_ratio[np.where(np.logical_and(0.5 < actual_ratio, actual_ratio <= 0.8))]) / total_num, 4),
        "80_110": round(len(actual_ratio[np.where(np.logical_and(0.8 < actual_ratio, actual_ratio <= 1.1))]) / total_num, 4),
        "110_150": round(len(actual_ratio[np.where(np.logical_and(1.1 < actual_ratio, actual_ratio <= 1.5))]) / total_num, 4),
        "150_200": round(len(actual_ratio[np.where(np.logical_and(1.5 < actual_ratio, actual_ratio <= 2))]) / total_num, 4),
        "200_300": round(len(actual_ratio[np.where(np.logical_and(2 < actual_ratio, actual_ratio <= 3))]) / total_num, 4),
        "300_500": round(len(actual_ratio[np.where(np.logical_and(3 < actual_ratio, actual_ratio <= 5))]) / total_num, 4),
        "500": round(len(actual_ratio[np.where(actual_ratio >= 5)]) / total_num, 4)
    }

    return dis, ratio_dict


def exp_distance_func(bandwidth_predictions=None, true_capacity=None, return_separate_item=False):
    # process nan value
    bandwidth_predictions[np.isnan(bandwidth_predictions)] = bandwidth_predictions[
        ~np.isnan(bandwidth_predictions)].mean()
    true_capacity[np.isnan(true_capacity)] = true_capacity[~np.isnan(true_capacity)].mean()
    true_capacity = true_capacity * 0.95

    x_norm = bandwidth_predictions / true_capacity
    x_norm = np.clip(x_norm, 0, 2.0)
    coef = np.ones_like(x_norm)
    coef = coef * 8.0
    coef[np.where(x_norm>1.0)] = coef[np.where(x_norm>1.0)] * 10.0
    coef[np.where(x_norm<0.8)] = coef[np.where(x_norm<0.8)] * 5.0
    params = -coef * (x_norm - 1.0) * (x_norm - 1.0)

    if return_separate_item:
        return {"10 * np.exp(params)": 10 * np.exp(params),}

    return 10 * np.exp(params)


def reward_func(bandwidth_predictions=None, true_capacity=None, return_separate_item=False):
    rewards = exp_distance_func(bandwidth_predictions,true_capacity, return_separate_item)
    return rewards


class MinMaxScalers:
    def __init__(self, device):
        self._torch_min = None
        self._torch_max = None
        self.device = device
        self.eps = 1e-3

    def fit_with_data(self, min_data, max_data):
        if len(min_data) == 0 or len(max_data) == 0:
            return

        self._torch_min = min_data
        self._torch_max = max_data

    def transform(self, x):
        x = torch.clip(((x - self._torch_min) / (self._torch_max - self._torch_min + self.eps) * 2.0 - 1.0), -1.0, 1.0)
        return x

    def reverse_transform(self, x):
        out = ((self._torch_max - self._torch_min + self.eps) * (x + 1.0) / 2.0) + self._torch_min
        placer = torch.zeros_like(out).to(self.device)
        out = torch.cat([out, placer], 1)
        out = torch.unsqueeze(out, 1)
        return out


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.y_label = collections.deque(maxlen=capacity)
        self.others = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def add_y_label(self, label):
        self.y_label.extend(label)

    def add_episode(self, episode):
        self.buffer.extend(episode)
    
    def add_others(self, other_qos):
        self.others.extend(other_qos)

    def np_array_sample(self, state_dim, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        transitions = np.array(transitions)
        state = transitions[:, :state_dim]
        action = transitions[:, state_dim]
        reward = transitions[:, state_dim+1]
        next_state = transitions[:, state_dim+2:-1]
        done = transitions[:, -1]
        return state, action, reward, next_state, done

    def get_pretrain_state_and_label(self, state_dim):
        transitions = np.array(self.buffer)
        return transitions[:, :state_dim], transitions[:, state_dim+2:-1]

    def get_state_and_label(self, state_dim):
        transitions = np.array(self.buffer)
        label = np.array(self.y_label)
        return transitions[:, :state_dim], label


obs_min = torch.tensor([6933.3335,0.,    0.,    0.,    0.,  693.3333
                            ,0.,    0.,    0.,    0.,    1.,    0.
                            ,0.,    0.,    0.,    1.,    0.,    0.
                            ,0.,    0.,   52.,    0.,    0.,    0.
                            ,0.,   52.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,-2187.5,    -2186.
                            , -2188.,-2188.,-2188.,-2179.5386, -2179.5386, -2179.818
                            , -2179.5,    -2179.5386, -1992.,-1992.,-1992.,-1992.
                            , -1992.,-1992.,-1992.,-1992.,-1992.,-1992.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.])
obs_max = torch.tensor([2.2961187e+08,2.2961187e+08,2.2961187e+08,2.2961187e+08,2.2576613e+08
                            ,2.5140640e+07,1.8429560e+07,2.5140640e+07,2.1878146e+07,2.1878146e+07
                            ,1.7480000e+03,1.7480000e+03,1.7480000e+03,1.7480000e+03,1.7190000e+03
                            ,1.9100000e+03,1.6560000e+03,1.9100000e+03,1.7060000e+03,1.7060000e+03
                            ,1.7220890e+06,1.7220890e+06,1.7220890e+06,1.7220890e+06,1.6932460e+06
                            ,1.8855480e+06,1.3822170e+06,1.8855480e+06,1.6408610e+06,1.6408610e+06
                            ,8.0290000e+03,8.0716841e+03,8.0716841e+03,7.1589414e+03,6.8999048e+03
                            ,8.0290000e+03,8.0716841e+03,7.6990000e+03,7.2922798e+03,7.1413486e+03
                            ,8.0270000e+03,8.0696841e+03,8.0696841e+03,7.1569414e+03,6.8979048e+03
                            ,8.0270000e+03,8.0696841e+03,7.6412500e+03,7.2892798e+03,7.0593486e+03
                            ,2.0000000e+02,2.0000000e+02,2.0000000e+02,2.0000000e+02,2.0000000e+02
                            ,2.0000000e+02,2.0000000e+02,2.0000000e+02,2.0000000e+02,2.0000000e+02
                            ,4.3364285e+02,3.8484000e+02,4.1659259e+02,1.4950000e+02,4.4811111e+02
                            ,4.3364285e+02,6.0400000e+02,8.3088892e+02,6.0400000e+02,4.1924139e+02
                            ,4.3907275e+03,4.0823064e+03,4.3304302e+03,4.3386089e+03,4.3304302e+03
                            ,4.7623037e+03,4.8302510e+03,4.6838955e+03,4.7108174e+03,4.6838955e+03
                            ,3.1400000e+03,3.0880000e+03,2.9200000e+03,3.0880000e+03,2.9860000e+03
                            ,3.1400000e+03,2.8910000e+03,3.1400000e+03,3.1400000e+03,3.0000000e+03
                            ,1.5530000e+03,1.5739735e+03,1.5530000e+03,1.5530000e+03,1.5530000e+03
                            ,1.5530000e+03,1.5530000e+03,1.5305000e+03,1.4965000e+03,1.5530000e+03
                            ,9.8529410e-01,9.8484850e-01,9.9009901e-01,9.8969072e-01,9.8969072e-01
                            ,9.8529410e-01,9.5717883e-01,9.7368419e-01,9.9000001e-01,9.9029124e-01
                            ,2.6875000e+02,3.7900000e+02,3.7900000e+02,1.5600000e+02,1.6000000e+02
                            ,2.6875000e+02,1.9000000e+02,1.6000000e+02,1.9000000e+02,1.6000000e+02
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00])

bwe_min = torch.tensor([5000.])
bwe_max = torch.tensor([8000000.])