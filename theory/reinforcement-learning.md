# Bellman Equations



## Reward

![equation](https://render.githubusercontent.com/render/math?math=R_%7Bt%7D%20%3D%20%5Csum_%7Bk%3D0%7D%5E%7B%5Cinf%7D%5Cgamma%5E%7Bk%7D%20r_%7Bt%2Bk%2B1%7D)

## Policy

Policy is a function Π(s, a) of the state and the current action. It returns the probability of taking action a in state s.



## State value function

![equation](https://render.githubusercontent.com/render/math?math=V%5E%7B%5Cpi%7D%28s%29%20%3D%20E_%7B%5Cpi%7D%5BR_%7Bt%7D%7Cs_%7Bt%7D%20%3D%20s%5D)

It is the expected return when starting from state s according to policy π



## Action value function

![equation](https://render.githubusercontent.com/render/math?math=Q%5E%7B%5Cpi%7D%28s%2Ca%29%20%3D%20E_%7B%5Cpi%7D%5BR_%7Bt%7D%7Cs_%7Bt%7D%3Ds%2C%20a_%7Bt%7D%3Da%5D)

It is the expected return given s and a under π



## Bellman equation for state value function 

![equation](https://render.githubusercontent.com/render/math?math=V%5E%7B%5Cpi%7D%28s%29%20%3D%20%5Csum_%7Ba%7D%5Cpi%28s%2Ca%29%5Csum_%7Bs%5E%7B%27%7D%7DP_%7Bss%5E%7B%27%7D%7D%5E%7Ba%7D%5BR_%7Bss%5E%7B%27%7D%7D%5E%7Ba%7D%2B%5Cgamma%20V%5E%7B%5Cpi%7D%28s%5E%7B%27%7D%29%5D)

## Bellman equation for action value function

![equation](https://render.githubusercontent.com/render/math?math=Q%5E%7B%5Cpi%7D%28s%2Ca%29%3D%5Csum_%7Bs%5E%7B%27%7D%7DP_%7Bss%5E%7B%27%7D%7D%5E%7Ba%7D%5BR_%7Bss%5E%7B%27%7D%7D%5E%7Ba%7D%2B%5Cgamma%20%5Csum_%7Ba%5E%7B%27%7D%7D%5Cpi%28s%5E%7B%27%7D%2Ca%5E%7B%27%7DQ%5E%7B%5Cpi%7D%28s%5E%7B%27%7D%2Ca%5E%7B%27%7D%29%29%5D)



# Training algorithms



## Q-learning



### Deterministic Bellman

![equation](https://render.githubusercontent.com/render/math?math=Q%28s%2Ca%29%20%3D%20r%20%2B%20%5Cgamma%20%2A%20max%28Q%28s%27%2Ca%27%29%29)

```python
# Randomize current QValues
rand_qvals = Q[state] + torch.rand(1,number_of_actions)/1000
# Get an action given the current QValues
action = torch.max(rand_qvals, 1)[1][0].item()
# Produce a new state given the chosen action
new_state, reward, done, info = env.step(action)
# Update QValues given current reward and QValue
Q[state, action] = reward + gamma * torch.max(Q[new_state])
state = new_state

```



### Stochastic Q-Learning

![equation](https://render.githubusercontent.com/render/math?math=Q%28s%2Ca%29%20%3D%20%281-%5Calpha%29%20Q%28s%2Ca%29%20%2B%20%5Calpha%5Br%20%2B%20%5Cgamma%20%2A%20max%28Q%28s%27%2Ca%27%29%29%5D)

```python
rand_qvals = Q[state] + torch.rand(1,number_of_actions)/1000
action = torch.max(rand_qvals, 1)[1][0].item()
new_state, reward, done, info = env.step(action)
Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + gamma * torch.max(Q[new_state]))
state = new_state
```



## E-Greedy decay

```python
random_for_egreedy = torch.rand(1)[0].item()
if random_for_egreedy > egreedy:
    rand_qvals = Q[state] + torch.rand(1,number_of_actions)/1000
    action = torch.max(rand_qvals, 1)[1][0].item()
else:
    action = env.action_space.sample()

if egreedy > egreedy_final:
    egreedy *= egreedy_decay

new_state, reward, done, info = env.step(action)
Q[state, action] = reward + gamma * torch.max(Q[new_state])
state = new_state
```



## Gradient descent - Neural Network

![equation](https://render.githubusercontent.com/render/math?math=Q%28s%2Ca%29%20%3D%20f%28s%2Ca%29)

f(.) is approximated using a neural network



### Standard optimization

```python
def optimize(self, state, action, new_state, reward, done):
	state = torch.Tensor(state).to(device)
	new_state = torch.Tensor(new_state).to(device)
	reward = torch.Tensor([reward]).to(device)

	if done:
		target_value = reward
	else:
		new_state_values = self.nn(new_state).detach()
		max_new_state_values = torch.max(new_state_values)
		target_value = reward + gamma * max_new_state_values

	predicted_value = self.nn(state)[action]
	loss = self.loss_func(predicted_value, target_value)
	self.optimizer.zero_grad()
	loss.backward()
        self.optimizer.step()
```



### Experience replay

```python
def optimize(self):

    if len(memory) < batch_size:
        return

    state, action, new_state, reward, done = memory.sample(batch_size)
    state = torch.Tensor(state).to(device)
    new_state = torch.Tensor(new_state).to(device)
    reward = torch.Tensor(reward).to(device)
    action = torch.LongTensor(action).to(device)
    done = torch.Tensor(done).to(device)

    new_state_values = self.nn(new_state).detach()
    max_new_state_values = torch.max(new_state_values, 1)[0]
    target_value = reward + (1 - done) * gamma * max_new_state_values

    predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
    loss = self.loss_func(predicted_value, target_value)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```



### Target Net

```python
def optimize(self):

    if len(memory) < batch_size:
        return

    state, action, new_state, reward, done = memory.sample(batch_size)
    state = torch.Tensor(state).to(device)
    new_state = torch.Tensor(new_state).to(device)
    reward = torch.Tensor(reward).to(device)
    action = torch.LongTensor(action).to(device)
    done = torch.Tensor(done).to(device)

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
```

