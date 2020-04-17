import numpy as np
import matplotlib.pyplot as plt
import math

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

class ElasticNetRegression(object):
    def __init__(self, _lambda = 0.0001, iterations=10000, alpha=0.1):             #Parent constructor
        self.iterations = iterations                                               #How many times will (w) get updated
        self.alpha = alpha                                                         #How much will w get changed with each step
        self.regularization = l2_regularization(_lambda = _lambda)                 #How much will we reduce the effect of a gradient step
         
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

elasticNetRegression = ElasticNetRegression()
elasticNetRegression.fit(X, y)

print('------------------Starting Training Error------------------')
print(elasticNetRegression.training_errors[0])
print('--------------------Final Training Error-------------------')
print(elasticNetRegression.training_errors[-1])
print('-----------------------Final Weights-----------------------')
print(elasticNetRegression.w)

#Create the plot
plt.scatter(my_data[:, 0].reshape(-1,1), y)
plt.title('(Multivariate) Elastic Net Regression')
plt.xlabel('sq feet')
plt.ylabel('price')
plt.plot(X, elasticNetRegression.predict(X))
