import csv
import random
import math
 
def loadCsv(filename):                             #Get filename
	lines = csv.reader(open(filename, "r"))         #Read file   
	dataset = list(lines)                           #Dataset is a list holding the lines of the file
	for i in range(len(dataset)):                   #For each row of the dataset
		dataset[i] = [float(x) for x in dataset[i]]    #For each feature of the dataset -> make it float
	return dataset                                  #Return the dataset
 
def splitDataset(dataset, splitRatio):         #Get dataset and percentage of split
	trainSize = int(len(dataset) * splitRatio)  #Clalculate ammount of rows going in the training set
	trainSet = []                               #Create an empty array holding the training data
	copy = list(dataset)                        #Copy the dataset as a list
	while len(trainSet) < trainSize:            #While the requested training size is not met
		index = random.randrange(len(copy))        #Get a random index from the dataset
		trainSet.append(copy.pop(index))           #Append the corresponding row in the training set and remove it from the dataset
	return [trainSet, copy]                     #Return training set and test set
 
def separateByClass(dataset):               #Get the dataset
	separated = {}                           #Create an empty set to hold the classes
	for i in range(len(dataset)):            #For each row of the dataset
		vector = dataset[i]                     #Vector is a row of the dataset
		if (vector[-1] not in separated):       #If the observation is not yet registered in classes set do it
			separated[vector[-1]] = []             #Get the observation
		separated[vector[-1]].append(vector)    #Append its values with it
	return separated                         #Return separated vector holding all observations by class
 
def mean(numbers):                          #Get attribute
	return sum(numbers)/float(len(numbers))  #Calculate the mean
 
def stdev(numbers):                         #Get attribute
	avg = mean(numbers)                      #Calculate mean
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)  #Claculate variance
	return math.sqrt(variance)               #Return the standard deviation
 
def summarize(dataset):           #Get dataset
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]   #For each class in the dataset get its mean and standard deviation
	del summaries[-1]                                
	return summaries               #Return final values for each class
 
def summarizeByClass(dataset):                         #Get the dataset
	separated = separateByClass(dataset)                #Get an array with all the different observations separated
	summaries = {}                                      #Create an empty list holding the summaries
	for classValue, instances in separated.items():     #For each different category get the values and how many items it has got inside
		summaries[classValue] = summarize(instances)       #Get the summary of each instance
	return summaries                                    #Return it
 
def calculateProbability(x, mean, stdev):                           #Get test row and corresponding mean and stdev
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2)))) #Clalculate the exponent
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent           #Return the probability of it belonging in the specified category
 
def calculateClassProbabilities(summaries, inputVector):      #Get the summaries and a test row
	probabilities = {}                                         #Create an empty set for the probabilities of each class
	for classValue, classSummaries in summaries.items():       #Get value and content of each summary
		probabilities[classValue] = 1                           #Initialize it to one
		for i in range(len(classSummaries)):                    #For each summary
			mean, stdev = classSummaries[i]                        #Get its mean and stdev
			x = inputVector[i]                                     #Get a row of the test set
			probabilities[classValue] *= calculateProbability(x, mean, stdev)  #Calculate the probabilities of a test row belonging to any category
	return probabilities                                       #Return the probabilities of each row of the test set
			
def predict(summaries, inputVector):                                    #Get the summaries and a test row
	probabilities = calculateClassProbabilities(summaries, inputVector)  #Get the array of the probabilities for each test element
	bestLabel, bestProb = None, -1                                       #Initialize variables holding the best probability
	for classValue, probability in probabilities.items():                #For each class
		if bestLabel is None or probability > bestProb:                     #If current probability is greate than the last
			bestProb = probability                                             #Assign the new best probability
			bestLabel = classValue                                             #Assign the new dominant category
	return bestLabel                                                     #Return the category that suits the test row the most
 
def getPredictions(summaries, testSet):         #Get the summaries of the instances of the training set and the test set
	predictions = []                             #Create an empty array for the predictions
	for i in range(len(testSet)):                #For eaach value in the test set
		result = predict(summaries, testSet[i])     #
		predictions.append(result)                  #Append the prediction to the predictions array
	return predictions                           #Return the array with the predictions
 
def getAccuracy(testSet, predictions):              #Get test set and the predictions this algorithm made for the test set
	correct = 0                                      #Counter for correct predictions
	for i in range(len(testSet)):                    #For each element in the test set
		if testSet[i][-1] == predictions[i]:            #If the prediction matches the observation
			correct += 1                                 #Increase the correct counter by one
	return (correct/float(len(testSet))) * 100.0     #Return the percentage of corrects
 
def main():
	filename = '00.iris-dataset.csv'  #file name
	splitRatio = 0.67                 #How many observations go to the training set
	dataset = loadCsv(filename)       #Load file into readable form
	trainingSet, testSet = splitDataset(dataset, splitRatio)  #Split dataset into training set and test set
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))  #0.Total number of rows 1.how many went to the training set 2.How many went to the test set
	summaries = summarizeByClass(trainingSet)          #Create the model
	predictions = getPredictions(summaries, testSet)   #Get the predictions based on this model
	accuracy = getAccuracy(testSet, predictions)       #Get the percentage of correct predictions
	print('Accuracy: {0}%').format(accuracy)           #Print the accuracy
 
main()