import numpy as np
import math

class LinearRegression(object):
    def __init__(self, iterations=10000, alpha=0.1):                               #Parent constructor
        self.iterations = iterations                                               #How many times will (w) get updated
        self.alpha = alpha                                                         #How much will w get changed with each step
         
    def fit(self, X, y):                                                           #Fit for basic regression X = array of features, y = vector of results 
        X=np.insert(X, 0, 1, axis=1)                                               #Add (y intercept) in first column starting with ones
        
        self.training_errors = []                                                  #With each update the new error will be added so that we can analyze the process
        
        limit= 1/math.sqrt(X.shape[1])                                             #A fancy way to generate a random number using the number of features of X
        self.w = np.array([np.random.uniform(-limit, limit, (X.shape[1]))])        #Create a vector (w) filled with (n) randomized weights
        
        for i in range(self.iterations):                                           #Start iterating
            y_pred = X @ self.w.T                                                  #Multiply each row of feature values of X with w to get predicted result y
            mse = np.mean(0.5 * (y - y_pred)**2)                                   #Calculate mean squared error
            mse = self.training_errors.append(mse)                                 #append error
            self.w -= self.alpha / len(X) * np.sum((X @ self.w.T - y) * X, axis=0) #Subtract old (w) with (learning_rate/len) times the partial derivative
            
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w.T)
        return y_pred
    
    
my_data = np.genfromtxt('01.4_column_data.csv', delimiter=',')
my_data = (my_data - my_data.mean())/(my_data.max() - my_data.min())  #feature scaling

X = my_data[:, :len(my_data[0]) - 1].reshape(-1, len(my_data[0]) - 1) # -1 tells numpy to figure out the dimension by itself
y = my_data[:, len(my_data[0])-1].reshape(-1,1)                       #Get y

linearRegression = LinearRegression()
linearRegression.fit(X, y)

print('------------------Starting Training Error------------------')
print(linearRegression.training_errors[0])
print('--------------------Final Training Error-------------------')
print(linearRegression.training_errors[-1])
print('-----------------------Final Weights-----------------------')
print(linearRegression.w)

#A plot for multivariate regression is not very helpfull
