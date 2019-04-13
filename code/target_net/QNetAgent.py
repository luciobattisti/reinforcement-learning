import torch
import torch.nn as nn
import torch.optim as optim
from NeuralNetwork import NeuralNetwork

class QNetAgent:

    def __init__(self, env, device, learning_rate, ninputs, hidden, noutputs):
        self.nn = NeuralNetwork(ninputs, hidden, noutputs)
        self.target_nn = NeuralNetwork(ninputs, hidden, noutputs)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        self.update_target_counter = 0
        self.env = env
        self.device = device

    def select_action(self, state, epsilon):
        random_for_egreedy = torch.rand(1).item()

        if random_for_egreedy > epsilon:
            with torch.no_grad():
                state = torch.Tensor(state).to(self.device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn, 0)[1].item()
        else:
            action = self.env.action_space.sample()

        return action

    def optimize(self, memory, batch_size, gamma, clip_error, update_target_frequency):

        if len(memory) < batch_size:
            return

        state, action, new_state, reward, done = memory.sample(batch_size)
        state = torch.Tensor(state).to(self.device)
        new_state = torch.Tensor(new_state).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        done = torch.Tensor(done).to(self.device)

        new_state_values = self.target_nn(new_state).detach()
        max_new_state_values = torch.max(new_state_values, 1)[0]
        target_value = reward + (1 - done) * gamma * max_new_state_values

        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        loss = self.loss_func(predicted_value, target_value)
        self.optimizer.zero_grad()
        loss.backward()
        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)
        self.optimizer.step()

        if self.update_target_counter % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

        self.update_target_counter + 1