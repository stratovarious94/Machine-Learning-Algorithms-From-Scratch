import numpy as np
import matplotlib.pyplot as plt
import math

class l1_regularization(): #Lasso Regression
    def __init__(self, _lambda):
        self._lambda = _lambda
        
    def __call__(self, w):
        return self._lambda * np.linalg.norm(w) #L1 normalization function (Sums the absolute value of all elements of (w)) and multiplies with (lambda)
        
    def grad(self, w):
        return self._lambda * np.sign(w)        #L1 normalization function derivative (_lambda * (plus/minus one | 0)) (Only sign survives this derivation)

class LassoRegression(object):
    def __init__(self, _lambda = 0.0001, iterations=10000, alpha=0.1):             #Parent constructor
        self.iterations = iterations                                               #How many times will (w) get updated
        self.alpha = alpha                                                         #How much will w get changed with each step
        self.regularization = l1_regularization(_lambda = _lambda)                 #How much will we reduce the effect of a gradient step
         
    def fit(self, X, y):                                                           #Fit for basic regression X = array of features, y = vector of results 
        X=np.insert(X, 0, 1, axis=1)                                               #Add (y intercept) in first column starting with ones
        
        self.training_errors = []                                                  #With each update the new error will be added so that we can analyze the process
        
        limit= 1/math.sqrt(X.shape[1])                                             #A fancy way to generate a random number using the number of features of X
        self.w = np.array([np.random.uniform(-limit, limit, (X.shape[1]))])        #Create a vector (w) filled with (n) randomized weights
        
        for i in range(self.iterations):                                           #Start iterating
            y_pred = X @ self.w.T                                                  #Multiply each row of feature values of X with w to get predicted result y
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))     #Calculate mean squared error + regularization
            mse = self.training_errors.append(mse)                                 #append error
            self.w -= self.alpha / len(X) * np.sum((X @ self.w.T - y) * X, axis=0)  + self.regularization.grad(self.w) #Subtract old (w) with (learning_rate/len) times the partial derivative
            
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w.T)
        return y_pred
    
my_data = np.genfromtxt('00.2_column_data.csv', delimiter=',')
my_data = (my_data - my_data.mean())/(my_data.max() - my_data.min()) #feature scaling

X = my_data[:, :len(my_data[0]) - 1].reshape(-1, len(my_data[0]) - 1) # -1 tells numpy to figure out the dimension by itself
y = my_data[:, len(my_data[0])-1].reshape(-1,1)                       #Get y

lassoRegression = LassoRegression()
lassoRegression.fit(X, y)

print('------------------Starting Training Error------------------')
print(lassoRegression.training_errors[0])
print('--------------------Final Training Error-------------------')
print(lassoRegression.training_errors[-1])
print('-----------------------Final Weights-----------------------')
print(lassoRegression.w)

#Create the plot
plt.scatter(my_data[:, 0].reshape(-1,1), y)
plt.title('(Multivariate) Lasso Regression')
plt.xlabel('sq feet')
plt.ylabel('price')
plt.plot(X, lassoRegression.predict(X))
