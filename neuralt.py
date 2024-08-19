from godot import exposed, export
from godot import *

import torch
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet_3h(nn.Module):
	def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
		super().__init__()
		self.linear1 = nn.Linear(input_size, hidden_size1)
		self.linear2 = nn.Linear(hidden_size1, hidden_size2)
		self.linear3 = nn.Linear(hidden_size2, output_size)

	def forward(self, x): #precition
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x

	def save(self, file_name='model_new.pth'):
		model_folder_path = './model'
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)

		file_name = os.path.join(model_folder_path, file_name)
		torch.save(self.state_dict(), file_name)
	

class Linear_QNet(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super().__init__()
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, output_size)
		self.linear = nn.Linear(hidden_size, output_size)

	def forward(self, x): #precition
		x = F.relu(self.linear1(x))
		x = self.linear2(x)
		return x

	def save(self, file_name='model2.pth'):
		model_folder_path = './model'
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)

		file_name = os.path.join(model_folder_path, file_name)
		torch.save(self.state_dict(), file_name)

class QTrainer:
	def __init__(self, model, lr, gamma):
		self.lr = lr
		self.gamma = gamma
		self.model = model
		self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #look into
		self.criterion = nn.MSELoss() #what are the options?

	def train_step(self, state, action, reward, next_state, done):
		state = torch.tensor(np.array(state), dtype=torch.float)
		next_state = torch.tensor(np.array(next_state), dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)
		reward = torch.tensor(reward, dtype=torch.float)
		# (n, x)

		if len(state.shape) == 1:
			# (1, x)
			state = torch.unsqueeze(state, 0)
			next_state = torch.unsqueeze(next_state, 0)
			action = torch.unsqueeze(action, 0)
			reward = torch.unsqueeze(reward, 0)
			done = (done, )

		# 1: predicted Q values with current state
		pred = self.model(state)

		# 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
		# pred.clone()
		# preds[argmax(action)] = Q_new
		#from my understaning the loss function is calculated here with reward, current and next state
		target = pred.clone()
		for idx in range(len(done)):
			Q_new = reward[idx]
			if not done[idx]:
				Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

			target[idx][torch.argmax(action[idx]).item()] = Q_new
		
		#applying loss function here
		self.optimizer.zero_grad()
		loss = self.criterion(target, pred)
		loss.backward()

		self.optimizer.step()

@exposed
class neuralt(Node):
	
	training = False
	best_model = True
	n_games = 0
	epsilon = 0 #randomness
	gamma = 0.9 #discount rate, play around <1, default 0.9
	LR = 0.001
	record = 0
	memory = deque(maxlen=10_000) #default 100_000
	#model = Linear_QNet(4, 128, 2) #input(state), hidden, output
	model = Linear_QNet_3h(4, 80, 40, 2)
	trainer = QTrainer(model, lr=LR, gamma=gamma)
	BATCH_SIZE = 64 #default 1000

	
	def _ready(self):
		print("model ready")
		if self.best_model:
			self.model.load_state_dict(torch.load('./model/model_3h.pth', weights_only=True))
	
	def train_short_memory(self, state_old, action, reward, state, done):
		action = [1,0] if action else [0,1]
		self.trainer.train_step(state_old, action, reward, state, done)
	
	#to do - train everything in batches maybe
	
	def train_long_memory(self):
		if len(self.memory) > self.BATCH_SIZE:
			mini_sample = random.sample(self.memory, self.BATCH_SIZE) #list of tuples
		else:
			mini_sample = self.memory
		
		states, actions, rewards, next_states, dones = zip(*mini_sample)
		self.trainer.train_step(states, actions, rewards, next_states, dones)
			
	def remember(self, state_old, action, reward, state, done):
		action = [1,0] if action else [0,1] #maybe will cause errors
		self.memory.append((state_old, action, reward, state, done)) # popleft if MAX_MEMORY is reached
		
	#play around with this maybe
	def get_action(self, state):
		# random moves: tradeoff exploration / exploitation
		action = 0
		if self.training:
			self.epsilon = 120 - self.n_games
		if random.randint(0, 200) < self.epsilon:
			action = 1 if random.randint(0,13) > 11 else 0
		else:
			state0 = torch.tensor(state, dtype=torch.float)
			prediction = self.model(state0)
			#print('prediction: ', prediction.item())
			action = 1 if torch.argmax(prediction).item() == 0 else 0
		
		return action
	
	def done(self, score):
		#train long memory
		self.n_games += 1
		#self.train_long_memory()
		
		if score > self.record:
			self.record = score
			if self.training:
				self.model.save()
		
		print('Game', self.n_games, 'Score', score, 'Record:', self.record)
		#print('Memory len:', len(self.memory))
		
		#maybe plot with matplotlib later

	def hello_world(self):
		print("hello world")
		
