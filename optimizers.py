import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
	def __init__(self):
		pass

	@abstractmethod
	def update(self):
		pass

class SGD(Optimizer):
	def __init__(self, layer, lrn_rate, reg_rate):
		self.layer = layer
		self.lrn_rate = lrn_rate
		self.reg_rate = reg_rate

	def update(self):
		for i in range(self.layer.num_params):
			self.layer.params[i] = (1 - self.lrn_rate * self.reg_rate) * self.layer.params[i] - self.lrn_rate * self.layer.grad_params[i]

class SGDMomentum(Optimizer):
	def __init__(self, layer, lrn_rate, friction):		
		self.lrn_rate = lrn_rate
		self.friction = friction
		self.layer = layer

		self.V = [np.zeros(layer.params[i].shape) for i in range(layer.num_params)]
		super().__init__()

	def update(self):
		for i in range(self.layer.num_params):
			self.V[i] = self.friction * self.V[i] + (1 - self.friction) * self.layer.grad_params[i]
			self.layer.params[i] -= self.lrn_rate * self.V[i]

class Adam(Optimizer):
	def __init__(self, layer, lrn_rate, beta1, beta2, epsilon):
		self.lrn_rate = lrn_rate
		self.iter_num = 1
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.layer = layer
		self.V = [np.zeros(layer.params[i].shape) for i in range(layer.num_params)]
		self.S = [np.zeros(layer.params[i].shape) for i in range(layer.num_params)]
		super().__init__()

	def update(self):
		for i in range(self.layer.num_params):
			self.V[i] = self.beta1 * self.V[i] + (1 - self.beta1) * self.layer.grad_params[i]
			self.S[i] = self.beta2 * self.S[i] + (1 - self.beta2) * self.layer.grad_params[i] ** 2
			corrV, corrS = self.V[i] / (1 - self.beta1 ** self.iter_num), self.S[i] / -(1 - self.beta2 ** self.iter_num)

			self.layer.params[i] -= self.lrn_rate * corrV / (self.epsilon + np.sqrt(corrS))
		self.iter_num += 1

def adamAll(model, lrn_rate, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8):
	curr_layer = model.in_layer.child

	while curr_layer:
		optimizer = Adam(curr_layer, lrn_rate, beta1, beta2, epsilon)
		curr_layer.set_optimizer(optimizer)
		curr_layer = curr_layer.child

def sgdAll(model, lrn_rate, reg_rate):
	curr_layer = model.in_layer.child

	while curr_layer:
		optimizer = SGD(curr_layer, lrn_rate, reg_rate)
		curr_layer.set_optimizer(optimizer)
		curr_layer = curr_layer.child

def sgdMomentumAll(model, lrn_rate, friction = 0.9):
	curr_layer = model.in_layer.child

	while curr_layer:
		optimizer = SGDMomentum(curr_layer, lrn_rate, friction)
		curr_layer.set_optimizer(optimizer)
		curr_layer = curr_layer.child