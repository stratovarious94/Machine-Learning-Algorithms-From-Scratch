import csv
import random
import math
import operator
 
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'r') as csvfile:              #Open file
	    lines = csv.reader(csvfile)                   #Tranform the file object into an array with each element representing a line
	    dataset = list(lines)                         #Dataset is a list of all lines
	    for x in range(len(dataset)-1):               #For each row of the dataset
	        for y in range(4):                          #For the first 4 columns
	            dataset[x][y] = float(dataset[x][y])      #Get the values and transform them to strings
	        if random.random() < split:                   #Randomly assign it based on our split threshold
	            trainingSet.append(dataset[x])            #In the training set
	        else:                                         #Or
	            testSet.append(dataset[x])                #The test test set
 
 
def euclideanDistance(instance1, instance2, length):
	distance = 0                                          #Initialize distance
	for x in range(length):                               #For each element in the test set
		distance += pow((instance1[x] - instance2[x]), 2)    #Calculate its square difference with the training set and add it to the distance
	return math.sqrt(distance)                            #Return the square of the distance
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []                             #Empty array holding the distances between each neighbor
	length = len(testInstance)-1               #Get the length of the test set
	for x in range(len(trainingSet)):            #For each element in the training set
		dist = euclideanDistance(testInstance, trainingSet[x], length)  #Calculate its euclidean distance from the rest
		distances.append((trainingSet[x], dist))    #And append a list holding the element and its distance in the distances array
	distances.sort(key=operator.itemgetter(1))   #Sort the distance array
	neighbors = []                               #Create an empty neighbors array
	for x in range(k):                           #For the given number of neighbors K
		neighbors.append(distances[x][0])           #Append the first k elements (The nearest elements)
	return neighbors                             #Return the k neighbors
 
def getResponse(neighbors):
	classVotes = {}                  #Initialize set holding all the different votes
	for x in range(len(neighbors)):  #For each neighbor
		response = neighbors[x][-1]     #Get the last distance
		if response in classVotes:      #If it already exists
			classVotes[response] += 1      #Increase them by 1
		else:
			classVotes[response] = 1       #Else create a new response starting at 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)  #Sort by most votes
	return sortedVotes[0][0]             #Return them
 
def getAccuracy(testSet, predictions):
	correct = 0                              #Counter for correct predictions
	for x in range(len(testSet)):            #For each element in the test set
		if testSet[x][-1] == predictions[x]:    #If the element's category is equal to the prediction
			correct += 1                           #Increase the counter
	return (correct/float(len(testSet))) * 100.0   #Return the percentage of correct predictions
	
def main():
	trainingSet=[]                                                                #Create an empty training set
	testSet=[]                                                                    #Create an empty test set
	split = 0.67                                                                  #Percentage that goes to the training set
	loadDataset('00.iris-dataset.csv', split, trainingSet, testSet)               #Transfolrm dataset into an array we can control
	print('Train set: ' + repr(len(trainingSet)))                                 #Print how much data went in the training set
	print('Test set: ' + repr(len(testSet)))                                      #Print how much data went in the test set
	predictions=[]                                                                #Create an empty array which will hold the predictions
	k = 3                                                                         #The number of neighbors 
	for x in range(len(testSet)):                                                 #For each element in the test set
		neighbors = getNeighbors(trainingSet, testSet[x], k)                         #Get the k nearest neighbors
		result = getResponse(neighbors)                                              #Find which category gets the most votes for this element (The more dominant)
		predictions.append(result)                                                   #Append prediction to the predictions array
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))    #And print first the predicted and then the actual value
	accuracy = getAccuracy(testSet, predictions)                                  #Get the percentage of predictions that are correct
	print('Accuracy: ' + repr(accuracy) + '%')                                    #Print it
	
main()

