import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

Transition = namedtuple('Transition', ('node_state', 'node_action', 'node_prob', 'reward'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)
    

class PPO(nn.Module):
    def __init__(self, input_dim, input_dim_train, output_dim, device, eps_clip=0.2):
        super(PPO, self).__init__()
        self.memory = Memory()
        self.eps_clip = eps_clip
        self.device = device

        self.fc1_train = nn.Linear(input_dim_train, 32)
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc_pi = nn.Linear(64, output_dim)
        
        self.s1 = nn.Linear(input_dim + 2, 64)
        self.s2 = nn.Linear(64, 32)
        self.s3 = nn.Linear(32, output_dim)
        
        self.layer_norm(self.fc1, std=1.0)
        self.layer_norm(self.fc2, std=1.0)
        self.layer_norm(self.fc_pi, std=0.01)
        
        self.layer_norm(self.s1, std=1.0)
        self.layer_norm(self.s2, std=1.0)
        self.layer_norm(self.s3, std=0.01)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        
    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    
    def pi(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = torch.tanh(x)

        x = self.fc_pi(x)
        prob = F.softmax(x, dim=1)

        return prob

    def pi_train(self, x):
        x = self.fc1_train(x)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = torch.tanh(x)

        x = self.fc_pi(x)
        prob = F.softmax(x, dim=1)

        return prob

    def pi_s(self, x):  
        x=self.s1(x)
        x = torch.tanh(x)

        x=self.s2(x)
        x = torch.tanh(x)
        
        x = self.s3(x)
        prob = F.softmax(x, dim=1)
        
        return prob
    
    def put_data(self, x):
        node_prob = x[2].gather(dim=1, index=torch.unsqueeze(x[1], dim=1))

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # trans_x = torch.transpose(x[0].cpu(), 0, 1).to(device)
        # new_x = (trans_x,) + x[1:]
        # self.memory.push(new_x[0], torch.unsqueeze(new_x[1], dim=1), node_prob, new_x[3])

        self.memory.push(x[0], torch.unsqueeze(x[1], dim=1), node_prob, x[3])

    def train_net(self, epochs=5, batch_size=256):
        batch = self.memory.sample()
        node_state, node_action, node_old_prob, reward = torch.cat(batch.node_state), torch.cat(batch.node_action), torch.cat(batch.node_prob),torch.cat(batch.reward)
        
        # sample from memory
        idx = np.random.choice(reward.size()[0], min(reward.size()[0], batch_size), replace=False)
        # idx = np.random.choice(node_state.size()[0], min(node_state.size()[0], batch_size), replace=False)
        node_state, node_action, node_old_prob, reward = node_state[idx], node_action[idx], node_old_prob[idx], reward[idx]

        for i in range(epochs):
            advantage = reward
            # node_new_prob = self.pi_train(node_state).gather(dim=1, index=node_action)
            node_new_prob = self.pi(node_state).gather(dim=1, index=node_action)
            
            ratio1 = torch.exp(torch.log(node_new_prob) - torch.log(node_old_prob))
            
            surr1 = ratio1 * advantage
            surr2 = torch.clamp(ratio1, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss1 = -torch.min(surr1, surr2).mean()
            
            loss = loss1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.lr_scheduler.step()

        self.memory = Memory()
