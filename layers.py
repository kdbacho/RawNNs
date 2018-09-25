import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
#Layers

class Layer(ABC):
	def __init__(self, params, parent):
		self.L = None
		self.optimizer = None
		self.params = params

		self.parent = parent
		self.child = None
		self.is_training = False

		if parent:
			parent.child = self

	def forward(self, x):
		self.L = self.ell(x)
		if not self.child:
			return self.L
		else:
			return self.child.forward(self.L)

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	@abstractmethod
	def ell(self, x):
		pass

	@abstractmethod
	def fast_back(self, grad_prev, L_curr):
		pass

	@abstractmethod
	def set_param_grads(self, grad_L, x, lrn_rate):
		pass

class Input(Layer):
	num_params = 0

	def __init__(self, size):
		self.in_size = size
		self.out_size = size
		super().__init__([], None)

	def ell(self, x):
		return x

	def fast_back(self, grad_prev, L_curr):
		pass

	def set_param_grads(self, grad_L, x):
		pass

class NonParametricLayer(Layer):
	num_params = 0

	def __init__(self, input_layer):
		self.out_size = input_layer.out_size
		self.in_size = self.out_size
		super().__init__([], input_layer)

	def set_param_grads(self, grad_L, x):
		pass

	@abstractmethod
	def ell(self, x):
		pass

	@abstractmethod
	def fast_back(self, grad_prev, L_curr):
		pass

class Softmax(NonParametricLayer):

	def __init__(self, input_layer):
		super().__init__(input_layer)

	def ell(self, x):
		exps = np.exp(x - np.max(x))
		return exps / np.sum(exps)

	def fast_back(self, grad_prev, L_curr):
		s = self.ell(L_curr).reshape(-1,1)
		jacobian = np.diagflat(s) - np.dot(s, s.T)
		return np.dot(jacobian.T, grad_prev)

class ReLU(NonParametricLayer):

	def __init__(self, input_layer):
		super().__init__(input_layer)

	def ell(self, x):
		return x * (x > 0)

	def drel(self, x):
		return 1 * (x > 0)

	def fast_back(self, grad_prev, L_curr):
		return self.drel(L_curr) * grad_prev

class PReLU(Layer):

	num_params = 1

	def __init__(self, input_layer):
		self.in_size = input_layer.out_size
		self.out_size = self.in_size

		params = [np.zeros((self.in_size,1))]
		super().__init__(params, input_layer)

	def ell(self, x):
		return x * (x > 0) + (self.params[0] * x) * (x <= 0)

	def dprel(self, x):
		return 1 * (x > 0) + self.params[0] * (x <= 0)

	def fast_back(self, grad_prev, L_curr):
		return self.dprel(L_curr) * grad_prev

	def set_param_grads(self, grad_L, x):
		temp = x * (x <= 0)
		self.grad_params = [np.dot(grad_L.T, temp)]

class Sigmoid(NonParametricLayer):

	def __init__(self, input_layer):
		super().__init__(input_layer)

	def ell(self, x):
		#x = np.clip(x, -500, 500)
		return 1.0/(1.0 + np.exp( -x ))

	def dsig(self, x):
		return self.ell(x) * (1 - self.ell(x))

	def fast_back(self, grad_prev, L_curr):
		return self.dsig(L_curr) * grad_prev

class Tanh(NonParametricLayer):
	def __init__(self, input_layer):
		super().__init__(input_layer)

	def ell(self, x):
		return np.tanh(x)

	def dtanh(self, x):
		return 1 - np.tanh(x) ** 2

	def fast_back(self, grad_prev, L_curr):
		return self.dtanh(L_curr) * grad_prev

class Affine(Layer):

	num_params = 2

	def __init__(self, in_layer, out_size):
		self.in_size = in_layer.out_size
		self.out_size = out_size
		W = np.random.randn(self.out_size, self.in_size) * np.sqrt(2.0 / self.in_size)
		b = np.zeros((self.out_size,1))
		
		params = [W,b]

		super().__init__(params, in_layer)

	def ell(self, x):
		return np.dot(self.params[0],x) + self.params[1]

	def fast_back(self, grad_prev, L_curr):
		return np.dot(self.params[0].T, grad_prev)

	def set_param_grads(self, grad_L, x):
		batch_size = grad_L.shape[1]
		self.grad_params = [np.dot(grad_L, x.T) / batch_size, grad_L.sum(1).reshape(self.out_size,1) / batch_size]

class Dropout(NonParametricLayer):
	def __init__(self, input_layer, p_survive):
		self.p_survive = p_survive
		super().__init__(input_layer)

	def ell(self, x):
		if not self.is_training:
			return x

		self.D = np.random.rand(x.shape[0], x.shape[1]) < self.p_survive
		return (self.D * x) / self.p_survive

	def fast_back(self, grad_prev, L_curr):
		return (self.D * grad_prev) / self.p_survive

class Conv2d(Layer):

	num_params = 2

	def __init__(self, in_layer, num_filters, kernel_size, stride = 1):
		self.in_size = self.in_layer.out_size
		self.out_size = 2
		self.stride = stride
		W = np.random.randn(kernel_size[0], kernel_size[1], self.in_size[2], num_filters)
		b = np.zeros(num_filters)

		params = [W, b]

		super().__init__(params, in_layer)

	def im2col(X, filter_shape, stride):
		b, m, n, c = X.shape
		sb, s0, s1, s2 = X.strides
	
		n_rows = m - filter_shape[0] + 1
		n_cols = n - filter_shape[1] + 1

		sh = b, n_rows, n_cols, c, filter_shape[0], filter_shape[1]
		strds = sb, s0, s1, s2, s0, s1

		out_view = np.lib.stride_tricks.as_strided(X, shape = sh, strides = strds)[:, ::stride, ::stride, :, :]
		return out_view.reshape(b, -1, filter_shape[0] * filter_shape[1] * c)

	def conv2d(X, F, stride):
		b = X.shape[0]
		n_f, h, w, c = F.shape
		A = im2col(X, (h, w), stride)
		F_flat = F.swapaxes(0,3).reshape(h * w * c, n_f)
		new_h = int((X.shape[1] - h) / stride + 1)
		new_w = int((X.shape[2] - w) / stride + 1)
		return np.dot(A, F_flat).reshape(b, new_h, new_w, n_f)


	def ell(self, x):
		return conv2d(x, self.params[0], self.stride) + self.params[1]

	def fast_back(self, grad_prev, L_curr):
		return conv2d(grad_prev, self.params[0], self.stride)

	def set_param_grads(self, grad_L, x):
		batch_size = x.shape[0]
		A = x.swapaxes(0,3)
		B = grad_L.swapaxes(0,2)

		self.grad_params[0] = (1 / batch_size) * conv2d(A,B).swapaxes(0,2)
		self.grad_params[1] = (1 / batch_size) * grad_L.sum(axis = (0,1,2))
		'''self.grad_params = [0,0]
		n_f = self.params[0].shape[0]
		self.grad_params[1] = np.array([np.sum(grad_L[:,:,f]) for f in range(n_f)])
		self.grad_params[0] = np.zeros(())

		for b in range(batch_size):
			for f in range(self.params[0].shape[0]):
				c = self.params[0][f].shape[2]
				self.grad_params[0][f] += np.stack([signal.correlate2d(x[b,:,:,k], grad_L[b,:,:,f]) for k in range(c)])
				self.grad_params[1][f] += np.sum(grad_L[b,:,:,f])
			self.grad_params[0][f] *= (1 / batch_size)
			self.grad_params[1][f] *= (1 / batch_size)'''


class Flatten(NonParametricLayer):
	def __init__(self, input_layer):
		super().__init__(input_layer)

	def ell(self, x):
		s = x.size
		return x.reshape((s,1))

	def fast_back(self, grad_prev, L_curr):
		return grad_prev.reshape(self.in_size)
