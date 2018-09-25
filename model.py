class Model:
	def __init__(self, in_layer, out_layer):
		self.in_layer = in_layer
		self.out_layer = out_layer

	def forward(self, X):
		return self.in_layer.forward(X)

	def predict(self, X):
		return np.argmax(self.forward(X), axis = 0)

	def enable_training(self):
		curr_layer = self.in_layer.child
		while curr_layer:
			curr_layer.is_training = True
			curr_layer = curr_layer.child

	def disable_training(self):
		curr_layer = self.in_layer.child
		while curr_layer:
			curr_layer.is_training = False
			curr_layer = curr_layer.child

	def backprop(self, x, y, batch_size):
		x = x.reshape(self.in_layer.in_size, batch_size)
		y = y.reshape(self.out_layer.out_size, batch_size)

		self.forward(x)
		epsilon = 1e-9

		grad_L_curr = (1 - y) / (1 - self.out_layer.L + epsilon) - y / (self.out_layer.L + epsilon)
		#grad_L_curr = - y / (self.out_layer.L + epsilon)
		self.out_layer.set_param_grads(grad_L_curr, self.out_layer.parent.L)

		prev = self.out_layer
		curr_layer = self.out_layer.parent
		
		while curr_layer.parent:
			grad_L_curr = prev.fast_back(grad_L_curr, curr_layer.L)
			curr_layer.set_param_grads(grad_L_curr, curr_layer.parent.L)
			prev = curr_layer
			curr_layer = curr_layer.parent

def test(model, test_inp, test_out):
	res = model.predict(test_inp)
	return np.count_nonzero(res == test_out) / test_out.size

def train(model, train_X, train_Y, train_labels, epochs, batch_size, report):
		model.enable_training()
		m = train_X.shape[1]

		for e in range(epochs):
			prog = 0
			perm = list(np.random.permutation(m))
			randomized_train_X = train_X[:, perm]
			randomized_train_Y = train_Y[:, perm]

			num_batches = m // batch_size

			for i in range(0, num_batches):
				batch_X = randomized_train_X[:, i * batch_size : (i + 1) * batch_size]
				batch_Y = randomized_train_Y[:, i * batch_size : (i + 1) * batch_size]
				model.backprop(batch_X, batch_Y, batch_size)

				curr_layer = model.in_layer.child
				while curr_layer:
					curr_layer.optimizer.update()
					curr_layer = curr_layer.child
				
				prog+=1
				if (not prog % 100) and report:
					print("Epoch ", e, " progress: ", prog , " / ", num_batches)

			print("Epoch", e, " ", test(model, train_X, train_labels))

		model.disable_training()
