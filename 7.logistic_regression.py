import numpy as np
import matplotlib.pyplot as plt
import math

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

class LogisticRegression():
    def __init__(self, iterations=10000, alpha = 0.1):      
        self.iterations = iterations              #How many times will (w) get updated                              
        self.alpha = alpha                        #alpha, how fast will it converge in each step
        self.sigmoid = Sigmoid()                  #Invoke Sigmoid() function

    def fit(self, X, y):
        limit= 1/math.sqrt(X.shape[1])                                                 #A fancy way to generate a random number using the number of features of X
        self.w = np.array([np.random.uniform(-limit, limit, (X.shape[1]))])            #Create a vector (w) filled with (n) randomized weights
        for i in range(self.iterations):                                               #Iterate
            y_pred = self.sigmoid(X @ self.w.T)                                        #Transform the predictions in values between 0 and 1
            self.w -= (self.alpha / len(X)) * np.sum((-(y - y_pred) * X), axis=0)      #(w) vector = (w) - (alpha)/(n_features) * sum(prediction_difference * array) 

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.w.T))).astype(int)
        return y_pred
    
my_data = np.genfromtxt('00.Social_Network_Ads.csv', delimiter=',') #Get data

X = my_data[1:, [2,3]].reshape(-1, 2)               #Get Age and Salary
y = my_data[1:, -1].reshape(-1,1)                   #Get last column (No need for feature scaling)

from sklearn.preprocessing import StandardScaler    #Feature Scaling with sklearn
scaler = StandardScaler()
X = scaler.fit_transform(X)

logisticRegression = LogisticRegression()
logisticRegression.fit(X, y)
print('-----------------------Final Weights-----------------------')
print(logisticRegression.w)
y_pred = logisticRegression.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

#Create the plot
from matplotlib.colors import ListedColormap                                     #Nice plot for classification
X_set, y_set = X, y.ravel()                                                      #Copy training set (Reduce dimensionality in y for a later purpose)
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),   #Used to generate coordinate matrices
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))   #It is essentially a matrix of equally spaced values 0 being the center


#Get the 2 grids, push X1 and X2 into 1 column each -> make them array -> get its transpose to make it vertical -> get prediction -> shape it as a coordinate matrix
plt.contourf(X1, X2, logisticRegression.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

arr1 = X_set[y_set == 0, 0]
arr2 = X_set[y_set == 0, 1]
#Plot the points (Scatterplot), 
for i, j in enumerate(np.unique(y_set)):  #For every unique values in the y set i, j get the values 0 and then 1
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)     #Apply coloring in dots and display legend

plt.title('Logistic Regression')
plt.xlabel('age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
