"""
Code altered from Chris Ketelsen's (PhD at CU) Machine Learning Course
"""


import argparse
import numpy as np 
import cPickle, gzip
import matplotlib.pyplot as plt

class Network:
    def __init__(self, sizes):
        self.L = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(n, 1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n, m) for (m,n) in zip(self.sizes[:-1], self.sizes[1:])]
        
    def g(self, z):
        """
        activation function 
        """
        return sigmoid(z)
    
    def g_prime(self, z):
        """
        derivative of activation function 
        """
        return sigmoid_prime(z) 
    
    def forward_prop(self, a):
        """
        memory aware forward propagation for testing 
        only.  back_prop implements it's own forward_prop 
        """
        for (W,b) in zip(self.weights, self.biases):
            a = self.g(np.dot(W, a) + b) 
        return a
    
    def gradC(self, a, y):
        """
        gradient of cost function 
        Assumes C(a,y) = (a-y)^2/2 
        """
        return (a - y)
    
    def SGD_train(self, train, epochs, eta, lam=0.0, verbose=True, bpverbose=False, test=None):
        """
        SGD for training parameters  
        epochs is the number of epocs to run 
        eta is the learning rate 
        lam is the regularization parameter 
        If verbose is set will print progressive accuracy updates 
        If test set is provided, routine will print accuracy on test set as learning evolves
        """
        n_train = len(train)
        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            for kk in range(n_train):
                xk = train[perm[kk]][0]
                yk = train[perm[kk]][1]
                dWs, dbs = self.back_prop(xk, yk, verbose=bpverbose)
                # TODO: Add L2-regularization 
                self.weights = [W - eta*(dW + lam*W) for (W, dW) in zip(self.weights, dWs)]
                self.biases = [b - eta*db for (b, db) in zip(self.biases, dbs)]
            if verbose:
                if epoch==0 or (epoch + 1) % 50 == 0:
                    acc_train = self.evaluate(train)
                    if test is not None:
                        acc_test = self.evaluate(test)
                        print "Epoch {:4d}: Train {:10.5f}, Test {:10.5f}".format(epoch+1, acc_train, acc_test) 
                    else:
                        print "Epoch {:4d}: Train {:10.5f}".format(epoch+1, acc_train)    
                
    def back_prop(self, x, y, verbose=False):
        """
        Back propagation for derivatives of C wrt parameters 
        """
        db_list = [np.zeros(b.shape) for b in self.biases]
        dW_list = [np.zeros(W.shape) for W in self.weights]
        
        a = x.reshape(len(x),1) 
        a_list = [a]
        z_list = [np.zeros(a.shape)] # Pad with throwaway so indices match 
        
        for W, b in zip(self.weights, self.biases):
            if(verbose):
                print("Wshape: %s, ashape: %s" % (W.shape,a.shape))
            z = np.dot(W, a) + b 
            z_list.append(z)
            a = self.g(z)
            a_list.append(a)
            
        # Back propagate deltas to compute derivatives 
        delta  = np.multiply(self.gradC(a,y.reshape(10,1)), self.g_prime(z_list[-1]))
        if(verbose): print("Delta-shape: %s, a-shape: %s, y-shape: %s, z-shape: %s" % (delta.shape, a.shape, y.shape, z_list[-1].shape))
        for ell in range(self.L-2,-1,-1):
            if(verbose):
                print("ell: %d, Delta-shape: %s, a^T-shape: %s" % (ell, delta.shape, a_list[ell].T.shape))
            db_list[ell] = delta
            dW_list[ell] = delta*a_list[ell].T
            delta = np.multiply(self.weights[ell].T.dot(delta), self.g_prime(z_list[ell]))
        return (dW_list, db_list)
    
    def evaluate(self, test):
        """
        Evaluate current model on labeled test data 
        """
        ctr = 0 
        for x, y in test:
            yhat = self.forward_prop(x)
            ctr += np.argmax(yhat) == np.argmax(y)
        return float(ctr) / float(len(test))
    
    def compute_cost(self, x, y):
        """
        Evaluate the cost function for a specified 
        training example. 
        """
        a = self.forward_prop(x)
        return 0.5*np.linalg.norm(a-y)**2
                
def sigmoid(z, threshold=20):
    z = np.clip(z, -threshold, threshold)
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,14))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname: 
	    plt.savefig(outname)
	else:
	    plt.show()

if __name__ == "__main__":

    f = gzip.open('../data/tinyTOY.pkl.gz', 'rb') # change path to ../data/tinyMNIST.pkl.gz after debugging
    train, test = cPickle.load(f) 

    nn = Network([2,30,2])
    nn.SGD_train(train, epochs=200, eta=0.25, lam=0.0, verbose=True, test=test)



