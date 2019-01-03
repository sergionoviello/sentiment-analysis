import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

nn_architecture = [
    {"input_dim": 2, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 128, "activation": "relu"},
    {"input_dim": 128, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 1, "activation": "sigmoid"}
]

## equivalent in Keras

# # Building a model
# model = Sequential()
# model.add(Dense(32, input_dim=2,activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])

# # Training
# history = model.fit(X_train, y_train, epochs=200, verbose=0)



def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        # create array of shape (layer_output_size, layer_input_size)
        # with random values in the range -1 +1
        # Weights values cannot be initialized with the same number because it leads to breaking symmetry problem.
        # Basically, if all weights are the same,
        # no matter what was the input X, all units in the hidden layer will be the same too
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1

        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
    return params_values


# forward propagation
# we use two functions, one for the single layer and onother for the entire network
# the formula is z[i] = W[i] * A[i-1] + b[i]
def single_layer_forward_prop(A_prev, W_curr, b_curr, activation='relu'):
  # calculation of the input value for the activation function
  Z_curr = np.dot(W_curr, A_prev) + b_curr

  # selection of activation function
  if activation is "relu":
      activation_func = relu
  elif activation is "sigmoid":
      activation_func = sigmoid
  else:
      raise Exception('Non-supported activation function')

  # return of calculated activation A and the intermediate Z matrix
  return activation_func(Z_curr), Z_curr


def full_forward_prop(X, params_values, nn_architecture):
  # creating a temporary memory to store the information needed for a backward step
  memory = {}
  # X vector is the activation for layer 0 
  A_curr = X

  # iteration over network layers
  for idx, layer in enumerate(nn_architecture):
      # we number network layers from 1
      layer_idx = idx + 1
      # transfer the activation from the previous iteration
      A_prev = A_curr

      # extraction of the activation function for the current layer
      activ_function_curr = layer["activation"]
      # extraction of W for the current layer
      W_curr = params_values["W" + str(layer_idx)]
      # extraction of b for the current layer
      b_curr = params_values["b" + str(layer_idx)]
      # calculation of activation for the current layer
      A_curr, Z_curr = single_layer_forward_prop(A_prev, W_curr, b_curr, activ_function_curr)

      # saving calculated values in the memory
      memory["A" + str(idx)] = A_prev
      memory["Z" + str(layer_idx)] = Z_curr

  # return of prediction vector and a dictionary containing intermediate values
  return A_curr, memory

# loss function
# the loss function is designed to show how far we are from the 'ideal' solution."
def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)

    return (Y_hat_ == Y).all(axis=0).mean()


#back prop
# We start by calculating a derivative of the cost function with respect to the prediction vector - result of forward propagation. This is quite trivial as it only consists of rewriting the following formula. Then iterate through the layers of the network starting from the end and calculate the derivatives with respect to all parameters according to the diagram shown in Figure 6. Ultimately, function returns a python dictionary containing the gradient we are looking for.

def full_backward_prop(Y_hat, Y, memory, params_values, nn_architecture):
  grads_values = {}

  # number of examples
  m = Y.shape[1]
  # a hack ensuring the same shape of the prediction vector and labels vector
  Y = Y.reshape(Y_hat.shape)

  # initiation of gradient descent algorithm
  dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));

  for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
    # we number network layers from 1
    layer_idx_curr = layer_idx_prev + 1

    # extraction of the activation function for the current layer
    activ_function_curr = layer["activation"]

    dA_curr = dA_prev

    A_prev = memory["A" + str(layer_idx_prev)]
    Z_curr = memory["Z" + str(layer_idx_curr)]

    W_curr = params_values["W" + str(layer_idx_curr)]
    b_curr = params_values["b" + str(layer_idx_curr)]

    # print('A_prev', A_prev.shape)
    # print('W_curr', W_curr.shape)
    # print('b_curr', b_curr.shape)
    # print('Z_curr', Z_curr.shape)

    dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

    grads_values["dW" + str(layer_idx_curr)] = dW_curr
    grads_values["db" + str(layer_idx_curr)] = db_curr

    # print('end layer--------', layer_idx_prev)

  return grads_values

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
  # number of examples
  m = A_prev.shape[1]

  # selection of activation function
  if activation is "relu":
      backward_activation_func = relu_backward
  elif activation is "sigmoid":
      backward_activation_func = sigmoid_backward
  else:
      raise Exception('Non-supported activation function')

  # calculation of the activation function derivative
  dZ_curr = backward_activation_func(dA_curr, Z_curr)

  # derivative of the matrix W
  dW_curr = np.dot(dZ_curr, A_prev.T) / m
  # derivative of the vector b
  db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
  # derivative of the matrix A_prev
  dA_prev = np.dot(W_curr.T, dZ_curr)

  # print('dA_prev', dA_prev.shape)
  # print('dW_curr', dW_curr.shape)
  # print('db_curr', db_curr.shape)
  # print('dZ_curr', dZ_curr.shape)

  return dA_prev, dW_curr, db_curr


def train(X, Y, nn_architecture, epochs, lr, verbose=False, callback=None):
  params_values = init_layers(nn_architecture, 2)
  cost_history = []
  accuracy_history = []

  for i in range(epochs):
    Y_hat, cache = full_forward_prop(X, params_values, nn_architecture)

    cost = get_cost_value(Y_hat, Y)
    cost_history.append(cost)

    accuracy = get_accuracy_value(Y_hat, Y)
    accuracy_history.append(accuracy)

    # print('acc', accuracy)

    # step backward - calculating gradient
    grads_values = full_backward_prop(Y_hat, Y, cache, params_values, nn_architecture)
    # updating model state
    params_values = update(params_values, grads_values, nn_architecture, lr)

    if(i % 50 == 0):
      if(verbose):
          print("Iteration: {} - cost: {} - accuracy: {:.5f}".format(i, cost, accuracy))
      if(callback is not None):
          callback(i, params_values)

  return params_values

def update(params_values, grads_values, nn_architecture, lr):
  for layer_idx, layer in enumerate(nn_architecture, 1):
    params_values["W" + str(layer_idx)] -= lr * grads_values["dW" + str(layer_idx)]
    params_values["b" + str(layer_idx)] -= lr * grads_values["db" + str(layer_idx)]
  return params_values

# activation functions
# To be able to go full circle and pass both forward and backward propagation, we also have to prepare their derivatives.

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;


#generate dataset
X, y = make_moons(n_samples = 1000, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print('X_train', X_train.shape)
y_trans = np.transpose(y_train.reshape((y_train.shape[0], 1)))
params = train(np.transpose(X_train), y_trans, nn_architecture, 10000, 0.001, True)

# preds, _ = full_forward_prop(np.transpose(X_test), params, nn_architecture)

# acc_test = get_accuracy_value(preds, np.transpose(y_test.reshape((y_test.shape[0], 1))))
# print("Test set accuracy: {:.2f}".format(acc_test))