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

import torch
import torch.nn as nn
from module import PretrainNet


class PretrainAgent(object):
    def __init__(self, input_dim=150, hidden_dim=256, fc_dim=256, output_dim=150, train_length=256,
                 device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim
        self.output_dim = output_dim
        self.train_length = train_length
        self.device = device

        self.agent = PretrainNet(input_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(1e6))
        self.loss_fn = nn.MSELoss()
        self.observation_scaler = None

    def set_scalar(self, scaler):
        self.observation_scaler = scaler

    def train(self, train_input, train_target):
        train_input = torch.tensor(train_input, dtype=torch.float).to(self.device)
        train_target = torch.tensor(train_target, dtype=torch.float).to(self.device)

        if self.observation_scaler:
            train_input = self.observation_scaler.transform(train_input)
            train_target = self.observation_scaler.transform(train_target)

        self.optimizer.zero_grad()
        out = self.agent(train_input)
        loss = self.loss_fn(out, train_target)
        loss.backward()
        self.optimizer.step()

    def evaluate(self, test_input, test_target):
        test_input = torch.tensor(test_input, dtype=torch.float).to(self.device)
        test_target = torch.tensor(test_target, dtype=torch.float).to(self.device)

        if self.observation_scaler:
            test_input = self.observation_scaler.transform(test_input)
            test_target = self.observation_scaler.transform(test_target)

        with torch.no_grad():
            predicts = self.agent(test_input)
            loss = self.loss_fn(predicts, test_target).detach().cpu().numpy()
        return loss
