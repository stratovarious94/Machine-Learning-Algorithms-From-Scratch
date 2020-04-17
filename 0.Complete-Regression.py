import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import combinations_with_replacement

def polynomial_features(X, degree):    #Get data and the degree in which we transform them
    n_samples, n_features = np.shape(X)

    def index_combinations():  #Create a combination of the features given for the specified degree
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

class Regression(object):
    iterations: float      #How many times will (w) get updated
    learning_rate: float   #How much will w get changed with each step
    def __init__(self, iterations, learning_rate): #Parent constructor
        self.iterations = iterations
        self.learning_rate = learning_rate
        
    def initialize_weights(self, n_features):
        limit= 1/math.sqrt(n_features)                                       #A fancy way to generate a random number
        self.w = np.array([np.random.uniform(-limit, limit, (n_features))])  #Create a vector (w) filled with (n) random numbers 
        
    def fit(self, X, y):                                                     #Fit for basic multivariate regression X = array of features, y = vector of results 
        X=np.insert(X, 0, 1, axis=1)                                         #Add (y intercept) in first column starting with ones
        self.training_errors = []                                            #With each update the new error will be added so that we can analyze the process
        self.initialize_weights(n_features = X.shape[1])                     #Create the (w) vector given the number of features from array (X) + (y intercept)
        for i in range(self.iterations):                                     #Start iterating
            y_pred = X @ self.w.T                                            #Multiply each row of feature values of X with w to get predicted result y
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))               #Calculate error
            mse = self.training_errors.append(mse)                                           #append error
            self.w -= self.learning_rate / len(X) * np.sum((X @ self.w.T - y) * X, axis=0) + self.regularization.grad(self.w)  #Subtract old (w) with (learning_rate/len) times the partial derivative
            
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w.T)
        return y_pred

#--------------------SIMPLE REGRESSIONS---------------------------------------------------------------------------------------------------------------------------------

class LinearRegression(Regression):
    iterations: float      #How many times will (w) get updated
    learning_rate: float   #How much will w get changed with each step
    gradient_descent: bool #True: use gradient descent, False: use batch optimization
    
    def __init__(self, iterations=1000, learning_rate=0.0001, gradient_descent=True): #Constructor from initializing hyperparameters also utilizing parent's constructor
        self.gradient_descent = gradient_descent
        self.regularization = lambda x: 0                                             #Regularization = 0 : Simple linear regression does not have regularization
        self.regularization.grad = lambda x: 0                                        #Regularization.grad = 0 : Simple linear regression does not have regularization
        super(LinearRegression, self).__init__(iterations = iterations, learning_rate = learning_rate)
        
    def fit(self, X, y):                        #Batch optimization
        if not self.gradient_descent:
            X = np.insert(X, 0, 1, axis=1)      #Insert (y intercept column)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)
            
class PolynomialRegression(Regression):
    def __init__(self, degree = 16, iterations=10000, learning_rate=0.01):
        self.degree = degree                          #Degree of polynomial function
        self.regularization = lambda x: 0             #No regularization
        self.regularization.grad = lambda x: 0        #No regularization
        super(PolynomialRegression, self).__init__(iterations=iterations,learning_rate=learning_rate)
        
    def fit(self, X, y):
        X = polynomial_features(X, degree = self.degree)
        super(PolynomialRegression, self).fit(X, y)
        
    def predict(self, X):
        X = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)
        
#--------------------LASSO REGRESSION-----------------------------------------------------------------------------------------------------------------------------------

class l1_regularization(): #Lasso Regression
    def __init__(self, _lambda):
        self._lambda = _lambda
        
    def __call__(self, w):
        return self._lambda * np.linalg.norm(w) #L1 normalization function (Sums the absolute value of all elements of (w)) and multiplies with (lambda)
        
    def grad(self, w):
        return self._lambda * np.sign(w)        #L1 normalization function derivative (_lambda * (plus/minus one | 0)) (Only sign survives this derivation)
    
class LassoRegression(Regression):
    def __init__(self, degree, _lambda, n_iterations=3000, learning_rate=0.01):
        self.degree = degree #Degree of polynomial function
        self.regularization = l1_regularization(alpha=_lambda)
        super(LassoRegression, self).__init__(n_iterations, learning_rate)
        
#--------------------RIDGE REGRESSION-----------------------------------------------------------------------------------------------------------------------------------
    
class l2_regularization(): #Ridge Regression
    def __init__(self, _lambda):
        self._lambda = _lambda
        
    def __call__(self, w):
        return self._lambda * 0.5 * w @ w.T     #L2 normalization function (Sums the square value of all elements of (w)) and multiplies with (lambda)
        
    def grad(self, w):
        return self._lambda * w                 #L2 normalization function derivative (_lambda * w) (Only w survives this derivation)
    
class RidgeRegression(Regression):
    def __init__(self, _lambda, n_iterations=1000, learning_rate=0.001):
        self.regularization = l2_regularization(alpha=_lambda) #Choose a lambda
        super(RidgeRegression, self).__init__(n_iterations, learning_rate)
        
class PolynomialRidgeRegression(Regression):
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01, gradient_descent=True):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        super(PolynomialRidgeRegression, self).__init__(n_iterations, learning_rate)
        
    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(PolynomialRidgeRegression, self).predict(X)
        
#--------------------ELASTIC NET REGRESSION-----------------------------------------------------------------------------------------------------------------------------
        
class l1_l2_regularization(): #Elastic Net Regression
    def __init__(self, _lambda, l1_ratio = 0.5):
        self._lambda = _lambda
        self.l1_ratio = l1_ratio #How much L1 normalization will affect the regression
        
    def __call__(self, w):
        l1_norm = self.l1_ratio * np.linalg.norm(w)
        l2_norm = (1 - self.l1_ratio) * 0.5 * w @ w.T 
        return self._lambda * (l1_norm + l2_norm)     #L1_L2 normalization function (Sums the square value of all elements of (w)) and multiplies with (lambda)
        
    def grad(self, w):
        l1_norm = self.l1_ratio * np.sign(w) 
        l2_norm = (1 - self.l1_ratio) * w
        return self._lambda * (l1_norm + l2_norm)     #L1_L2 normalization function derivative (_lambda * w) (Only w survives this derivation)
    
class ElasticNet(Regression):
    def __init__(self, degree=1, reg_factor=0.05, l1_ratio=0.5, n_iterations=3000, 
                learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_l2_regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        super(ElasticNet, self).__init__(n_iterations, learning_rate)
        
    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(ElasticNet, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(ElasticNet, self).predict(X)



my_data = np.genfromtxt('00.2_column_data.csv', delimiter=',')
my_data = (my_data - my_data.mean())/(my_data.max() - my_data.min()) #feature scaling

X = my_data[:, 0].reshape(-1,1)                                      #Get X
y = my_data[:, 1].reshape(-1,1)                                      #Get y

polynomialRegression = PolynomialRegression()
polynomialRegression.fit(X, y)

print('------------------Starting Training Error------------------')
print(polynomialRegression.training_errors[0])
print('--------------------Final Training Error-------------------')
print(polynomialRegression.training_errors[-1])
print('-----------------------Final Weights-----------------------')
print(polynomialRegression.w)

#Create the plot
plt.scatter(my_data[:, 0].reshape(-1,1), y)
plt.title('Plynomial Linear Regression')
plt.xlabel('sq feet')
plt.ylabel('price')
plt.plot(X, polynomialRegression.predict(X))






