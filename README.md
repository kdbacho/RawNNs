# RawNNs
A Machine Learning library written from scratch using NumPy along with a guide.
## Example usage
To illustrate how the library functions, we will train a simple mnist classifier.
First we make the right imports
```python
import numpy as np
from optimizers import sgdAll
from layers import Input, Affine, ReLU, Sigmoid
from model import Model, train, test
```
Although the library does not use Tensorflow, we will use it in this example to load the mnist dataset. This allows you to test this example on your own machine without having to manually download any datasets (though you will need Tensorflow). We now load the data into numpy arrays

```python
import tensorflow as tf

def one_hot(x,b):
	return np.array([0 if i != x else 1 for i in range(b)])

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

trainable_labels = np.array([one_hot(x, 10) for x in train_labels])

eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
```

Our classifier will have will have 4 layers with sizes 784, 200, 50 and 10. The first two activations will be ReLUs while the last is a sigmoid. The first argument of each layer is always the previous.

```python
X = Input(784)
h = Affine(X, 200)
h = ReLU(h)
h = Affine(h, 50)
h = ReLU(h)
h = Affine(h, 10)
Y = Sigmoid(h)

mnist_classifier = Model(in_layer = X, out_layer = Y)
```
Although we can set an optimizer for each layer, we simply attach a SGD optimizer with regularization to all of the layers

```python
reg_rate = 5.0 / train_data.shape[0]
sgdAll(mnistc, 0.005, reg_rate)
```

We now train

```python
train(mnistc, train_data.T, trainable_labels.T, train_labels, 20, 25, False)
```
and after training we output our results on the test set

```python
print("Final Test: ", test(mnistc, eval_data.T, eval_labels))
```
