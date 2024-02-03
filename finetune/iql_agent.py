import torch
import copy
import module


def asymmetric_l2_loss(diff, expectile=0.7):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IQL(object):
    def __init__(
        self,
        state_dim=150,
        hidden_dim=256,
        action_dim=1,
        expectile=0.7,
        discount=0.99,
        tau=0.005,
        temperature=3.0,
        device="cpu",
        is_fine_tuned=False,
        multiple_output_num=5
    ):
        self.state_dim = state_dim
        self.device = device
        self.actor = module.Actor(state_dim, hidden_dim, multiple_output_num).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=int(1e6))

        self.critic = module.Critic(state_dim, hidden_dim, action_dim, multiple_output_num).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.value = module.ValueCritic(state_dim, hidden_dim, multiple_output_num).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.temperature = temperature

        self.expectile = expectile
        self.is_fine_tuned = is_fine_tuned

        self.observation_scaler = None
        self.action_scaler = None
        self.reward_scaler = None

    def load_pretrain_encoder(self, state_encoder):
        self.actor.encoder.load_state_dict(state_encoder.state_dict())
        for params in self.actor.encoder.parameters():
            params.requires_grad = False

        self.critic.encoder.load_state_dict(state_encoder.state_dict())
        for params in self.critic.encoder.parameters():
            params.requires_grad = False

        self.value.encoder.load_state_dict(state_encoder.state_dict())
        for params in self.value.encoder.parameters():
            params.requires_grad = False

    def update_scaler(self, observation_scaler, action_scaler, reward_scaler=None):
        self.observation_scaler = observation_scaler
        self.action_scaler = action_scaler
        self.reward_scaler = reward_scaler

    def freeze_actor(self):
        for params in self.actor.parameters():
            params.requires_grad = False

    def unfreeze_actor(self):
        for params in self.actor.parameters():
            params.requires_grad = True

        if self.is_fine_tuned:
            for params in self.actor.encoder.parameters():
                params.requires_grad = False

    def take_best_action(self, state, hid_s, cell_s):
        if self.observation_scaler:
            state = self.observation_scaler.transform(state)

        # make sure state shape is (batch_size, state_dim)
        action = self.actor.forward(state)

        if self.action_scaler:
            action = self.action_scaler.reverse_transform(action)
        return action, hid_s, cell_s

    def update_v(self, states, actions):
        with torch.no_grad():
            q1, q2 = self.critic_target(states, actions)
            q = torch.minimum(q1, q2).detach()

        v = self.value(states)
        value_loss = asymmetric_l2_loss(q - v, self.expectile).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def update_q(self, states, actions, rewards, next_states, done):
        with torch.no_grad():
            next_v = self.value(next_states)
            target_q = (rewards + self.discount * (1-done) * next_v).detach()

        q1, q2 = self.critic(states, actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_actor(self, states, actions):
        with torch.no_grad():
            v = self.value(states)
            q1, q2 = self.critic_target(states, actions)
            q = torch.minimum(q1, q2)
            exp_a = torch.exp((q - v) * self.temperature)
            exp_a = torch.clamp(exp_a, max=100.0).squeeze(-1).detach()

        mu = self.actor(states)
        actor_loss = (exp_a.unsqueeze(-1) * ((mu - actions)**2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

    def train(self, transition_dict):
        # Sample replay buffer
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        if self.observation_scaler:
            states = self.observation_scaler.transform(states)
            next_states = self.observation_scaler.transform(next_states)

        if self.action_scaler:
            actions = self.action_scaler.transform(actions)

        if self.reward_scaler:
            rewards = self.reward_scaler.transform(rewards)

        # Update
        self.update_v(states, actions)
        self.update_actor(states, actions)
        self.update_q(states, actions, rewards, next_states, dones)
        self.update_target()

    def save_policy(self, fname):
        def _func(x: torch.Tensor, hid_s, cell_s):
            # x shape is (1, 1, 150)
            # should squeeze to (1, 150)
            x = torch.squeeze(x, dim=1)

            best_action, hid_out, cell_out = self.take_best_action(x, hid_s, cell_s)

            # out shape (1, 1, 2)

            return best_action, hid_out, cell_out

        dummy_x = torch.rand(1, 1, self.state_dim).to(self.device)
        dummy_hidden_state = torch.rand(1, 1).to(self.device)
        dummy_cell_state = torch.rand(1, 1).to(self.device)

        self.freeze_actor()

        traced_script = torch.jit.trace(_func, (dummy_x, dummy_hidden_state, dummy_cell_state), check_trace=False)

        if fname.endswith(".onnx"):
            torch.onnx.export(
                traced_script,
                (dummy_x, dummy_hidden_state, dummy_cell_state),
                fname,
                export_params=True,
                opset_version=11,
                input_names=["obs", "hidden_states", "cell_states"],
                output_names=["output", "state_out", "cell_out"],
            )

        self.unfreeze_actor()