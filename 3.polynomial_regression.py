import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

def polynomial_features(X, degree):          #Get data and the degree in which we transform them
    n_samples, n_features = np.shape(X)      #n_samples = columns

    def index_combinations():                #Create a combination of the features given for the specified degree
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]  #Create the combinations object
        flat_combs = [item for sublist in combs for item in sublist]                                 #Append into a list
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)             #Number of possible combinations
    X_new = np.empty((n_samples, n_output_features))
    
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new

class PolynomialRegression(object):
    def __init__(self, degree=16, iterations=10000, alpha=0.1):                     #Parent constructor
        self.degree = degree                                                       #Degree of polynomial function
        self.iterations = iterations                                               #How many times will (w) get updated
        self.alpha = alpha                                                         #How much will w get changed with each step
        
         
    def fit(self, X, y):                                                           #Fit for basic multivariate regression X = array of features, y = vector of results 
        X = polynomial_features(X, degree = self.degree)
        X=np.insert(X, 0, 1, axis=1)
        
        self.training_errors = []                                                  #With each update the new error will be added so that we can analyze the process
        
        limit= 1/math.sqrt(X.shape[1])                                             #A fancy way to generate a random number using the number of features of X
        self.w = np.array([np.random.uniform(-limit, limit, (X.shape[1]))])        #Create a vector (w) filled with (n) randomized weights
        print(self.w)
        for i in range(self.iterations):                                           #Start iterating
            y_pred = X @ self.w.T                                                  #Multiply each row of feature values of X with w to get predicted result y
            mse = np.mean(0.5 * (y - y_pred)**2)                                   #Calculate mean squared error
            mse = self.training_errors.append(mse)                                 #append error
            self.w -= self.alpha / len(X) * np.sum((X @ self.w.T - y) * X, axis=0) #Subtract old (w) with (learning_rate/len) times the partial derivative
            
    def predict(self, X):
        X = polynomial_features(X, degree=self.degree)
        X=np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w.T)
        return y_pred
    
    
my_data = np.genfromtxt('00.2_column_data.csv', delimiter=',')
my_data = (my_data - my_data.mean())/(my_data.max() - my_data.min()) #feature scaling

X = my_data[:, :len(my_data[0]) - 1].reshape(-1, len(my_data[0]) - 1) # -1 tells numpy to figure out the dimension by itself
y = my_data[:, len(my_data[0])-1].reshape(-1,1)                       #Get y

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
