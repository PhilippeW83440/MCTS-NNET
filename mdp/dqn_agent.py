import numpy as np
import random
from collections import namedtuple, deque
import pdb
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import logging
import utils_nn as utils
from datetime import datetime

import torch.onnx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e4)
BATCH_SIZE = 32 # 64
TARGET_UPDATE = 10000 # update target network every ... 1000 looks OK as well
LR = 5e-4
LR = 2.5e-4


class ReplayBuffer:
	def __init__(self, buffer_size, batch_size, seed):
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
		self.seed = random.seed(seed)

	def add(self, state, action, reward, next_state, done):
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		experiences = random.sample(self.memory, k=self.batch_size)

		states, actions, rewards, next_states, dones = [], [], [], [], []
		for e in experiences:
			states.append(e.state)
			actions.append(e.action)
			rewards.append(e.reward)
			next_states.append(e.next_state)
			dones.append(e.done)
		states = torch.from_numpy(np.array(states)).float().to(device)
		actions = torch.from_numpy(np.array(actions)).long().to(device)
		rewards = torch.from_numpy(np.array(rewards)).float().to(device)
		next_states = torch.from_numpy(np.array(next_states)).float().to(device)
		dones = torch.from_numpy(np.array(dones)).float().to(device)

		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		return len(self.memory)

class DNN(nn.Module):
	def __init__(self, inputs=44, outputs=5):
		super(DNN, self).__init__()
		nfc = 200
		self.fc1 = nn.Linear(inputs, nfc)
		self.fc2 = nn.Linear(nfc, nfc)
		self.fc3 = nn.Linear(nfc, outputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class CNN(nn.Module):
	def __init__(self, inputs=44, outputs=5):
		super(CNN, self).__init__()
		nflat = 6000 # 14720 # 3072
		nfilters = 64
		nfc = 64
		# in_channels, out_channels, kernel_size, stride
		self.conv1 = nn.Conv2d( 1,  nfilters,  (3,3), stride=1, padding=1)
		self.conv2 = nn.Conv2d( nfilters, nfilters,  (3,3), stride=1, padding=1)
		self.conv3 = nn.Conv2d( nfilters, nfilters,  (3,3), stride=1, padding=1)
		self.conv4 = nn.Conv2d( nfilters, nfilters,  (3,3), stride=1, padding=1)
		self.conv5 = nn.Conv2d( nfilters, nfilters,  (3,3), stride=1, padding=1)
		self.conv6 = nn.Conv2d( nfilters, nfilters,  (3,3), stride=1, padding=1)

		self.conv7 = nn.Conv2d( nfilters, 3,  (1,1)) # reduce dimensionality

		self.fc1 = nn.Linear(1+nflat, nfc)
		self.fc2 = nn.Linear(nfc, nfc)
		self.fc3 = nn.Linear(nfc, outputs)

		self.pipo = nn.Linear(inputs, outputs)

	def forward(self, x): # [32, 200, 100]
		#if x.shape[0] > 1:
		#	pdb.set_trace()
		#else:
		#	x = x.view(x.shape[0], -1) # [1, 200*100]
		#	x = self.pipo(x)
		#	return x
		vego = x[:, 0, 0]
		#x[:, 0, 0] = 0
		x = x.unsqueeze(1) # [N=32, Cin=1, 200, 100]

		x = F.relu(self.conv1(x)) # [32, 64, ..., ...]
		x = F.relu(self.conv2(x)) # [32, 64, ..., ...]
		x = F.relu(self.conv3(x)) # [32, 64, ..., ...]
		x = F.relu(self.conv4(x)) # [32, 64, ..., ...]
		x = F.relu(self.conv5(x)) # [32, 64, ..., ...]
		x = F.relu(self.conv6(x)) # [32, 64, ..., ...]
		x = F.relu(self.conv7(x)) # [32, 64, ..., ...]

		x = x.view(x.shape[0], -1) # [32, 3072]
		enc = torch.cat((vego.unsqueeze(1), x), 1) # [32, 3073]

		x = F.relu(self.fc1(enc))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

		#x = inputs[:, 4:]  # [4, 40]
		#x = x.unsqueeze(1) # [N=4, Cin=1, L=40]
		#x = F.relu(self.bn1(self.conv1(x))) # [N, 32, 10]
		#x = F.relu(self.bn2(self.conv2(x))) # [N, 32, 10]
		#x = self.maxpool(x) # [N, 32, 1]

		#x = x.view(x.shape[0], -1) # [N, 32]
		#ego = inputs[:, 0:4] # [N, 4]
		#enc = torch.cat((ego, x), 1) # [N, 36]

		#x = F.relu(self.fc1(enc))
		#x = F.relu(self.fc2(x))
		#x = self.fc3(x)
		return x


class CNNold(nn.Module):
	def __init__(self, inputs=44, outputs=5):
		super(CNN, self).__init__()
		nfilters = 64
		nfc = 64
		# in_channels, out_channels, kernel_size, stride
		self.conv1 = nn.Conv1d( 1, nfilters,  4, stride=4)
		self.bn1 = nn.BatchNorm1d(nfilters)
		self.conv2 = nn.Conv1d(nfilters, nfilters, 1, stride=1)
		self.bn2 = nn.BatchNorm1d(nfilters)
		self.maxpool = nn.MaxPool1d(10)
		self.fc1 = nn.Linear(4+nfilters, nfc)
		self.fc2 = nn.Linear(nfc, nfc)
		self.fc3 = nn.Linear(nfc, outputs)

	def forward(self, inputs):
		#pdb.set_trace()
		x = inputs[:, 4:]  # [4, 40]
		x = x.unsqueeze(1) # [N=4, Cin=1, L=40]
		x = F.relu(self.bn1(self.conv1(x))) # [N, 32, 10]
		x = F.relu(self.bn2(self.conv2(x))) # [N, 32, 10]
		x = self.maxpool(x) # [N, 32, 1]

		x = x.view(x.shape[0], -1) # [N, 32]
		ego = inputs[:, 0:4] # [N, 4]
		enc = torch.cat((ego, x), 1) # [N, 36]

		x = F.relu(self.fc1(enc))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x



class DqnAgent():
	def __init__(self, mdp, args, seed, ddqn=True, model_state='reduce'): # Double DQN feature
		self.mdp = mdp
		self.model_state = model_state
		if model_state == 'reduce':
			# R7: ego_y, ego_vy, ttc, {car_x, car_y, car_vx, car_vy}smallest_ttc
			self.state_size = 7 # mdp.state_size()
			self.model_state = mdp.reduce_state
		elif model_state == 'project':
			# R8: ego speed + 3 closest (y_cross, t_cross) crossing points on the ego path
			# Car Trajectories are projected on the ego path (and everything is normalized)
			self.state_size = 8 # mdp.state_size()
			self.model_state = mdp.project_state
		elif model_state == 'grid':
			self.state_size = 200*10 # mdp.state_size()
			self.model_state = mdp.grid_state
		self.action_size = mdp.action_size()
		self.gamma = mdp.discount()
		self.seed = random.seed(seed)
		self.iters = 0
		self.args = args
		self.ddqn = ddqn

		# Q-Network
		if (args.restore is not None and "cnn" in args.restore) or args.nn == 'cnn':
			self.dqn_local = CNN(self.state_size, self.action_size).to(device)
			self.dqn_target = CNN(self.state_size, self.action_size).to(device)
		else: # default to dnn
			self.dqn_local = DNN(self.state_size, self.action_size).to(device)
			self.dqn_target = DNN(self.state_size, self.action_size).to(device)
		self.optimizer = optim.Adam(self.dqn_local.parameters(), lr=LR)

		if args.restore is not None:
			restore_path = os.path.join('models/', args.restore + '.pth.tar')
			logging.info("Restoring parameters from {}".format(restore_path))
			utils.load_checkpoint(restore_path, self.dqn_local, self.optimizer)
			self.dqn_target.load_state_dict(self.dqn_local.state_dict())
			self.saveOnnx(mdp.startState(), args.restore)

		# Replay memory
		self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
		self.t_step = 0

	# return loss
	def step(self, state, action, r, next_state, done):
		s = self.model_state(state)
		a = self.mdp.actionToIndex(action)
		sp = self.model_state(next_state)
		self.memory.add(s, a, r, sp, done)
		if len(self.memory) > BATCH_SIZE:
			experiences = self.memory.sample()
			return self.learn(experiences)
		else:
			return None

	# ACHTUNG: act returns action INDEX !!!
	def act(self, s, eps=0.):
		state = self.model_state(s)
		# tuple to np.array to torch, use float, add batch dim and move to gpu
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.dqn_local.eval()
		with torch.no_grad():
			qvalues = self.dqn_local(state)
		self.dqn_local.train()

		if random.random() > eps:
			qval = qvalues.cpu().numpy()
			#print(qval)
			a = np.argmax(qval)
		else:
			a = random.choice(np.arange(self.action_size))
		return self.mdp.indexToAction(a)

	def learn(self, experiences):
		states, actions, rewards, next_states, dones = experiences

		# unsqueeze to get [batch_dim, 1], then squeeze to get back to a [batch_dim] vector
		Q_expected = self.dqn_local(states).gather(1, actions.unsqueeze(1)).squeeze(1)

		# Double DQN implementation: retrieve argmax a' in next_states with local_network 
		if self.ddqn:
			# Double DQN implementation: retrieve argmax a' in next_states with local_network 
			actionsp = torch.max(self.dqn_local(next_states),1)[1] # retrieve argmax
			Q_targets_next = self.dqn_target(next_states).gather(1, actionsp.unsqueeze(1)).squeeze(1).detach()
		else:
			# Standard DQN implementation
			Q_targets_next = self.dqn_target(next_states).max(1)[0].detach()

		Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

		loss = F.mse_loss(Q_expected, Q_targets)

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		# clip loss ??? clip params ???
		# for param in dqn_local.parameters():
		# 	param.grad.data.clamp_(-1, 1)
		# Below we clip gradient (as done in previous pytorch predictions project)
		a = torch.nn.utils.clip_grad_norm_(self.dqn_local.parameters(), 10)
		self.optimizer.step()

		self.iters += 1
		if self.iters % TARGET_UPDATE == 0:
			self.dqn_target.load_state_dict(self.dqn_local.state_dict())

		return loss

	def save(self, num_epoch, mean_score, is_best=True):
		now = datetime.now()
		dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
		filename = self.args.nn+'Date'+dt_string+'Epoch'+str(num_epoch)+'Score'+"{:.2f}".format(mean_score)
		logging.info("Save model {} with mean_score {}".format(filename, mean_score))
		utils.save_checkpoint({'num_epoch': num_epoch,
								'state_dict': self.dqn_local.state_dict(),
								'optim_dict' : self.optimizer.state_dict(),
								'mean_score': mean_score},
								is_best = True,
								checkpoint = 'models/',
								filename=filename)

	def getV(self, s):
		state = self.model_state(s)
		# unsqueeze to add the BatchDim (1 in this case)
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.dqn_local.eval()
		with torch.no_grad():
			qvalues = self.dqn_local(state)
		qvalues = qvalues.cpu().numpy() # [BatchDim=1, 5]
		Vs = np.max(qvalues[0,:])
		return Vs

	def getQ(self, s, a):
		state = self.model_state(s)
		action = self.mdp.actionToIndex(a)
		# unsqueeze to add the BatchDim (1 in this case)
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.dqn_local.eval()
		with torch.no_grad():
			qvalues = self.dqn_local(state)
		qvalues = qvalues.cpu().numpy() # [BatchDim=1, 5]
		Qsa = qvalues[0,action]
		return Qsa

	def saveOnnx(self, s, name):
		state = self.model_state(s)
		# unsqueeze to add the BatchDim (1 in this case)
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		torch.onnx.export(self.dqn_local, state, 'models/'+name+".onnx", verbose=True)
