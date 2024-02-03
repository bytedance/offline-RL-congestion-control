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


class Actor(nn.Module):
    def __init__(self, observation_shape, hidden_dim, multiple_output_num):
        super().__init__()
        self.encoder = StateEncoder(observation_shape, hidden_dim)
        self.decoder = MultiOutputDecoder(hidden_dim, hidden_dim, multiple_output_num)
        self.ac = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        out = self.ac(x)
        return out


class Critic(nn.Module):
    def __init__(self, observation_shape, hidden_dim, action_dim, multiple_output_num=0):
        super().__init__()
        self.encoder = StateEncoder(observation_shape, hidden_dim)
        self.q1 = MultiOutputDecoder(hidden_dim + action_dim, hidden_dim, multiple_output_num)
        self.q2 = MultiOutputDecoder(hidden_dim + action_dim, hidden_dim, multiple_output_num)


    def forward(self, state, action):
        state = self.encoder(state)
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2


class ValueCritic(nn.Module):
    def __init__(self, observation_shape, hidden_dim, multiple_output_num=0):
        super().__init__()
        self.encoder = StateEncoder(observation_shape, hidden_dim)
        self.decoder = MultiOutputDecoder(hidden_dim, hidden_dim, multiple_output_num)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CustomEncoder(nn.Module):
    def __init__(self, observation_shape, hidden_dim, action_dim, finetune_layer=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc_head = nn.Linear(observation_shape, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.ac1 = nn.LeakyReLU()
        self.fc_body = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.block = ResidualBlock(self.hidden_dim, self.hidden_dim)
        self.finetune_layer = finetune_layer
        if self.finetune_layer:
            self.fc_ft = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc_head(x)
        x = self.layer_norm(x)
        x_out = self.ac1(x)
        x_out = self.fc_body(x_out)
        x_out = self.ac1(x_out)
        x_out = self.block(x_out)
        if self.finetune_layer:
            x_out = self.fc_ft(x_out)
        x_out = self.output(x_out)
        return x_out


class StateEncoder(nn.Module):
    def __init__(self, observation_shape, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc_head = nn.Linear(observation_shape, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.ac1 = nn.LeakyReLU()
        self.block = ResidualBlock(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        x = self.fc_head(x)
        x = self.layer_norm(x)
        x = self.ac1(x)
        x = self.block(x)
        return x


class StateDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.ac1 = nn.LeakyReLU()
        self.block = ResidualBlock(self.hidden_dim, self.hidden_dim)
        self.fc_tail = nn.Linear(hidden_dim, self.output_dim)
        self.ac2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.ac1(x)
        x = self.block(x)
        x = self.fc_tail(x)
        x = self.ac2(x)
        return x


class MultiOutputDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fc_head = nn.Linear(input_dim, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.ac1 = nn.LeakyReLU()

        # output
        self.fc_body = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.block = ResidualBlock(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, output_dim)

        # weight
        self.weight_fc_body = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.weight_block = ResidualBlock(self.hidden_dim, self.hidden_dim)
        self.weight_output = nn.Linear(self.hidden_dim, output_dim)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        # (batch_size, state_dim)
        x = self.fc_head(x)
        x = self.layer_norm(x)
        x = self.ac1(x)
        # output
        x_out = self.fc_body(x)
        x_out = self.ac1(x_out)
        x_out = self.block(x_out)
        x_out = self.output(x_out)

        #weight
        weight_out = self.weight_fc_body(x)
        weight_out = self.ac1(weight_out)
        weight_out = self.weight_block(weight_out)
        weight_out = self.weight_output(weight_out)
        weight_out = self.soft_max(weight_out)

        # x_out is in the shape of (batch_size, output_dim), weight_out is also in the shape of (batch_size, output_dim)
        # use torch.view to implement batch matrix-matrix product, the output will be (batch_size, 1, 1)
        output = torch.bmm(x_out.view(-1, 1, self.output_dim), weight_out.view(-1, self.output_dim, 1))
        output = torch.squeeze(output, dim=2)

        # (batch_size, 1)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.output_size = output_size
        self.block = nn.Sequential(
            nn.Linear(input_size, self.output_size),
            nn.LayerNorm(self.output_size),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class PretrainNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.encoder = StateEncoder(state_dim, hidden_dim)
        self.decoder = StateDecoder(hidden_dim, state_dim)
        self.ac = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.ac(x)
        return x
