import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

trainingdata = np.load("train.mat")
X = trainingdata#[0:1000]

label = []
for line in open("train.labels"):
    lbl = int(line)
    label.append(lbl)
y = label#[0:1000]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def f(x):
    '''
    I used tanh because it goes [-1,1] unlike sigmoid
    '''
    #return 1 / (1 + np.exp(-x))
    return np.tanh(x)

def f_deriv(x):
    #return f(x) * (1 - f(x))
    return 1.0 - np.tanh(x)**2

# Initialize parameters to random values
def setup_and_init_weights(nn_structure):
    '''Basically from the github code'''
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l], 1))
    return W, b

def init_tri_values(nn_structure):
    '''Basically from the github code'''
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l], 1))
    return tri_W, tri_b

def feed_forward(x, W, b):
    '''Basically from the github code'''
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise, 
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        #node_in_p = node_in.to_numpy().reshape(node_in.shape[0], 1)
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)  
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l)) 
    return h, z

def calculate_out_layer_delta(y, h_out, z_out):
    '''Modified from the github code'''
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    #return -(y - h_out) * f_deriv(z_out)
    return np.multiply(-(y - h_out), f_deriv(z_out))

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    '''Modified from the github code'''
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    #return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)
    return np.multiply(np.dot(w_l.T, delta_plus_1), f_deriv(z_l))

def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):
    '''Altered the github code to turn it into a binary classifier'''
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting to Train Neural Network for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%250 == 0 or cnt==(iter_num-1):
            print('\nIteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            #print("sample: {}".format(i))
            h, z = feed_forward(X[i].reshape(X[i].shape[0],1), W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i] - np.sum(h[l])))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    #tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis])) 
                    tri_W[l] += np.dot(delta[l+1], h[l].T)
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += np.sum(delta[l+1])
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
        #print("X",end="")
    return W, b, avg_cost_func


# Assess model performance
def predict_y(W, b, X, n_layers):
    '''Modified from the github code to become a binary predictor with an output of +1 or -1'''
    m = X.shape[0]
    y = np.zeros((m,1))
    for i in range(m):
        #h, z = feed_forward(X[i, :], W, b)
        h, z = feed_forward(X[i].reshape(X[i].shape[0],1), W, b)
        #y[i] = np.argmax(h[n_layers])
        #print(h[n_layers])
        y[i] = 1 if h[n_layers] >= 0 else -1
        #print(y[i] == y_test[i])
    return y

print("Begin Training.")
nn_structure = [X.shape[1], 50, 1]
W, b, avg_cost_func = train_nn(nn_structure, X, y)
print("Training Complete.")

X_test = np.load("test.mat")

print("Beginning Predictions.")
y_pred = predict_y(W, b, X_test, len(nn_structure))
#print(y_pred)
print ("Predictions Complete.")
#print(accuracy_score(y_test, y_pred)*100)

with open("results.txt", "w") as resultsfile:
    for y in y_pred:
        if y == 1:
            resultsfile.write("+1\n")
            #print("+1", file=resultsfile)
        else:
            resultsfile.write("-1\n")
            #print("-1", file=resultsfile)

print("Predictions stored in results.txt")