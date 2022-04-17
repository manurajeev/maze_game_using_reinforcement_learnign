import numpy as np
import time
import math

class NeuralNetwork:
    def __init__(self, neurons, activation='relu'):
        self.neurons = neurons
        
        # Choose activation function
        if activation == 'sigmoid':
            self.activation = self.sigmoid
        elif activation == 'tanh':
            self.activation = self.tanh
        elif activation == 'relu':
            self.activation = self.relu
 
        self.params = self.initialize()
        # Save all linear and activation caches
        self.cache = {}
        self.cache_target = {}

    #Activation Functions
    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        else:
            return 1/(1 + np.exp(-x))

    def tanh(self, x, derivative=False):
        hyperbolic = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        if derivative:
            return (1 - hyperbolic(x) * hyperbolic(x))
        else:
            return hyperbolic
         
    def relu(self, x, derivative=False):
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        else:
            return np.maximum(0, x)

    
    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def initialize(self):
        input_layer = self.neurons[0]
        hidden_layer = self.neurons[1]
        output_layer = self.neurons[2]
        
        params = {
            "weights1": 0.01 * np.random.randn(input_layer, hidden_layer),
            "bias1": 0.01 * np.zeros((1, hidden_layer)),
            "weights2": 0.01 * np.random.randn(hidden_layer, output_layer),
            "bias2": 0.01 * np.zeros((1, output_layer))
        }
        return params
    

    def forward(self, x):
        # self.cache["inp"] = x
        # self.cache["linear1"] = np.matmul(self.params["weights1"], self.cache["inp"].T) + self.params["bias1"]
        # self.cache["act1"] = self.activation(self.cache["linear1"])
        # self.cache["linear2"] = np.matmul(self.params["weights2"], self.cache["act1"]) + self.params["bias2"]

        self.cache["inp"] = x
        self.cache["linear1"] = np.dot(self.cache["inp"],self.params["weights1"]) + self.params["bias1"]
        self.cache["act1"] = self.activation(self.cache["linear1"])
        self.cache["linear2"] = np.dot(self.cache["act1"],self.params["weights2"]) + self.params["bias2"]
        #self.cache["act2"] = self.activation(self.cache["linear2"])
        return self.cache["linear2"]

    def forward_target(self, x):
        # self.cache["inp"] = x
        # self.cache["linear1"] = np.matmul(self.params["weights1"], self.cache["inp"].T) + self.params["bias1"]
        # self.cache["act1"] = self.activation(self.cache["linear1"])
        # self.cache["linear2"] = np.matmul(self.params["weights2"], self.cache["act1"]) + self.params["bias2"]

        self.cache_target["inp"] = x
        self.cache_target["linear1"] = np.dot(self.cache["inp"],self.params["weights1"]) + self.params["bias1"]
        self.cache_target["act1"] = self.activation(self.cache["linear1"])
        self.cache_target["linear2"] = np.dot(self.cache["act1"],self.params["weights2"]) + self.params["bias2"]
        #self.cache["act2"] = self.activation(self.cache["linear2"])
        return self.cache_target["linear2"]
    
    #backpropagate function
    def backward(self, y, output):
        '''
        m = y.shape[0]
        
        dZ2 = output - y
        dW2 = (1./m) * np.dot(dZ2, self.cache["act1"].T)
        db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.params["weights2"].T, dZ2)
        dZ1 = dA1 * self.activation(self.cache["linear1"], derivative=True)
        dW1 = (1./m) * np.dot(dZ1, self.cache["inp"])
        db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"weights1": dW1, "bias1": db1, "weights2": dW2, "bias2": db2}
        return self.grads
        '''
        m = y.shape[0]
        
        dZ2 = output - y
        dW2 = (1./m) * np.dot(self.cache["act1"].T, dZ2)
        db2 = (1./m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.params["weights2"].T)
        dZ1 = dA1 * self.activation(self.cache["linear1"], derivative=True)
        dW1 = (1./m) * np.dot(self.cache["inp"].T, dZ1)
        db1 = (1./m) * np.sum(dZ1, axis=0, keepdims=True)

        self.grads = {"weights1": dW1, "bias1": db1, "weights2": dW2, "bias2": db2}
        return self.grads
    

    def calculateLoss(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def cost_function(self, output, y):
        #m = Y.shape[1]
        cost = (np.square(output - y)).mean(axis=None)
        return cost

    

    def optimize(self, l_rate=0.01, beta=.9):
        '''
            Stochatic Gradient Descent (SGD):
            θ^(t+1) <- θ^t - η∇L(y, ŷ)

            Adam:
            
        '''
        #if self.optimizer == "sgd":
        for key in self.params:
            self.params[key] = self.params[key] - l_rate * self.grads[key]
        '''
        elif self.optimizer == "adam":
            #Implement adam
            for key in self.params:
                key
        '''

    
    '''
    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))

    def train(self, x_train, y_train, x_test, y_test, epochs=10, 
              batch_size=64, optimizer='sgd', l_rate=0.1, beta=.9):
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)
        
        # Initialize optimizer
        self.optimizer = optimizer
        
        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"
        
        # Train
        for i in range(self.epochs):
            # Shuffle
            
            #permutation = np.random.permutation(x_train.shape[0])
            #x_train_shuffled = x_train[permutation]
            #y_train_shuffled = y_train[permutation]
            
            x_train = np.array(x_train)
            y_train = np.array(y_train)

            for j in range(num_batches):
                # Batch
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = np.array(x_train[begin:end])
                y = np.array(y_train[begin:end])
                
                # Forward
                output = self.feed_forward(x)
                # Backprop
                _ = self.backpropagate(y, output)
                # Optimize
                self.optimize(l_rate=l_rate, beta=beta)
                #print(j)

            # Evaluate performance
            # Training data
            output = self.feed_forward(x_train)
            train_acc = self.accuracy(y_train, output) 
            train_loss = self.cross_entropy_loss(y_train, output)
            # Test data
            output = self.feed_forward(x_test)
            test_acc = self.accuracy(y_test, output)
            test_loss = self.cross_entropy_loss(y_test, output)
            print(template.format(i+1, time.time()-start_time, train_acc, train_loss, test_acc, test_loss))

    
    def model(self, state, epochs=10, 
              batch_size=64, optimizer='sgd', l_rate=0.1, beta=.9):
        #forward, backward, optimize
        print()
    '''

