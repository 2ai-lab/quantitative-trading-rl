import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TransformerQNetwork, QNetwork
from collections import deque
import torch.optim as optim
import random


class DQN:
    def __init__(self, input_size, output_size, learning_rate, gamma, num_layers=3, num_heads=3, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # simple q network
        self.q_network = QNetwork(input_size, output_size)
        self.target_network = QNetwork(input_size, output_size)
        
        # transformer based q network
        self.num_layers = num_layers
        self.num_heads = num_heads
        # self.q_network = TransformerQNetwork(input_size, output_size, feature_size=input_size, num_heads=num_heads, num_layers=self.num_layers).to(self.device)
        # self.target_network = TransformerQNetwork(input_size, output_size, feature_size=input_size, num_heads=num_heads, num_layers=self.num_layers).to(self.device)
        
        self.memory = deque(maxlen=10000)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=learning_rate)
        
        # use when transformer q-networks 
        # self.criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        
        # use when simple q-networks are used
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.loss = []
        self.accuracy = []
        self.batch_size = batch_size
        self.action_size = output_size
        self.test_accuracy = []
        self.test_loss = []

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    
    def remember(self, state, action, reward, next_state, done, labels):
        self.memory.append((state, action, reward, next_state, done, labels))

    def decide_action(self, state, epsilon):
        if random.random() > epsilon:  # Exploit: select the action with max Q-value.
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state).to(self.device)
                return q_values.argmax().item()
        else:  # Explore: select a random action.
            return random.randrange(self.action_size)
    
    
    def train(self, state, action, reward, next_state, done, labels, count):
        if len(self.memory) < self.batch_size:
            return
        self.optimizer.zero_grad()
        minibatch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done, labels = zip(*minibatch)

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)

        # next_state = F.log_softmax(next_state, dim=1)
        # state = F.log_softmax(state, dim=1)
        q_values = self.q_network(state).to(self.device)
        next_q_values = self.target_network(next_state).detach()
        # print(f'Next QVShape: {next_q_values.shape} NextQV: {next_q_values}')
        # print(f'Labels: {labels} and Shape: {labels.shape}')
        target_q_values = reward + (1 - done) * self.gamma * next_q_values.max(1)[0]

        # loss = self.criterion(q_values.gather(1, action.view(-1, 1)), target_q_values.view(-1, 1))
        # print(f'Action: {action}, TargetQV-Rounded: {torch.round(target_q_values)}')
        log_labels = F.log_softmax(labels, dim=0)
        log_target = F.log_softmax(torch.round(target_q_values), dim=0)
        # print(log_labels, log_target)
        loss = self.criterion(log_labels, log_target)
        # print(f'Loss: {loss}')
        # loss = self.criterion(torch.argmax(q_values), torch.round(target_q_values))
    # print(f'Q-Values: {q_values}, Action: {action}')
        # print(f'Q-Values: {q_values}, Action: {action}, TargetQValues- Rounded: {torch.round(target_q_values)}, Action-View: {action.view(-1, 1)}, QV-shape: {q_values.shape}')
        # print(f'1st condition: {q_values.gather(1, action.view(-1, 1))}')
        # print(f'2nd condition: {target_q_values}')
        # print(f'1st part: {(q_values.gather(1, action.view(-1, 1)) == target_q_values.view(-1, 1))}')
        # print(f'2nd part: {(q_values.gather(1, action.view(-1, 1)) == target_q_values.view(-1, 1)).float()}')
        # print(f'3rd part: {(q_values.gather(1, action.view(-1, 1)) == target_q_values.view(-1, 1)).float().mean()}')
        # accuracy = (q_values.gather(1, action.view(-1, 1)) == target_q_values.view(-1, 1)).sum().item() / self.batch_size
        # accuracy = (q_values.gather(1, action.view(-1, 1)) == target_q_values.view(-1, 1)).float().mean().item()
        accuracy = (labels == torch.round(target_q_values)).float().mean().item()
        # print(f'Accuracy: {accuracy}')
        self.accuracy.append(accuracy)
        loss.requires_grad = True
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss.item())

    def test(self, state, action, reward, next_state, done, labels, count):
        if len(self.memory) < self.batch_size:
            return
        self.optimizer.zero_grad()
        minibatch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done, labels = zip(*minibatch)

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)

        # next_state = F.log_softmax(next_state, dim=1)
        # state = F.log_softmax(state, dim=1)
        q_values = self.q_network(state).to(self.device)
        next_q_values = self.target_network(next_state).detach()
        # print(f'Next QVShape: {next_q_values.shape} NextQV: {next_q_values}')
        # print(f'Labels: {labels} and Shape: {labels.shape}')
        target_q_values = reward + (1 - done) * self.gamma * next_q_values.max(1)[0]

        # loss = self.criterion(q_values.gather(1, action.view(-1, 1)), target_q_values.view(-1, 1))
        # print(f'Action: {action}, TargetQV-Rounded: {torch.round(target_q_values)}')
        log_labels = F.log_softmax(labels, dim=0)
        log_target = F.log_softmax(torch.round(target_q_values), dim=0)
        # print(log_labels, log_target)
        loss = self.criterion(log_labels, log_target)
        self.test_loss.append(loss.item())
        self.test_accuracy.append((labels == torch.round(target_q_values)).float().mean().item())

        # print(f'Loss: {loss}')
        # loss = self.criterion(torch.argmax(q_values), torch.round(target_q_values))

        # print(f'Q-Values: {q_values}, Action: {action}')