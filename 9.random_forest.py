from random import seed
from random import randrange
from csv import reader
from math import sqrt
 
def load_csv(filename):         #Load a CSV file
	file = open(filename, "r")   #Open file
	lines = reader(file)         #Save each line in an array
	dataset = list(lines)        #Tranform the dataset into list
	return dataset               #Return it 
 
def str_column_to_float(dataset, column):       #Convert string column to float
	for row in dataset:                          #Go through the dataset
		row[column] = float(row[column].strip())  #If there is a string in thw row, remove it
 
def str_column_to_int(dataset, column):             #Get dataset and column
	class_values = [row[column] for row in dataset]  #For each row get the value of the corespoding column
	unique = set(class_values)                       #Copy them as a set so that they are unique
	lookup = dict()                                  #Create an empty dictionary
	for i, value in enumerate(unique):               #For each value in unique get the index and the value
		lookup[value] = i                             #Perform horizontal lookup
	for row in dataset:                              #For each row in the dataset
		row[column] = lookup[row[column]]             #Replace column value with its lookup value
	return lookup                                    #Return results
 
def cross_validation_split(dataset, n_folds):#Split a dataset into (n_folds) folds
	dataset_split = list()                    #Create an array holding each split
	dataset_copy = list(dataset)              #Create a copy of the dataset
	fold_size = int(len(dataset) / n_folds)   #Size of each fold = datasset size / (n_folds)
	for i in range(n_folds):                     #For each fold
		fold = list()                             #Create a list to hold a fold
		while len(fold) < fold_size:                #While fold array has not met the required size
			index = randrange(len(dataset_copy))     #Get a random element from the copied dataset and its index
			fold.append(dataset_copy.pop(index))     #Use the index to append element into fold array and remove it from the dataset
		dataset_split.append(fold)              #Append the fold array inside the dataset_split array
	return dataset_split                     #Ultimately we return dataset_split list, which is a list holding fold sets (Validation set)
 
def accuracy_metric(actual, predicted):  #Calculate accuracy percentage. "Get arrays with actual and predicted values
	correct = 0                           #Initiate accuracy counter
	for i in range(len(actual)):          #For each element in observations
		if actual[i] == predicted[i]:      #If the prediction is correct
			correct += 1                    #Increase counter by 1
	return correct / float(len(actual)) * 100.0  #Return percentage of correct predictions
 
def evaluate_algorithm(dataset, algorithm, n_folds, *args):   #Get dataset, a tree building algorithm, and number of folds for cross validation
	folds = cross_validation_split(dataset, n_folds)           #Create an array with all the folds of the dataset (Validation set)
	scores = list()                                            #Create the list holding the scores of each iteration of the algorithm (Scores set)
	for fold in folds:                                         #For each fold (fold is a subarray of the dataset)
		train_set = list(folds)                                 #Copy the validation set in training_set
		train_set.remove(fold)                                  #And remove the current fold (The subarray)
		train_set = sum(train_set, [])                          #Sum all the elements excluding the current fold
		test_set = list()                                       #Initialize a test set
		for row in fold:                                          #For each row of the fold
			row_copy = list(row)                                   #Copy the row
			test_set.append(row_copy)                              #Append it in the test_set
			row_copy[-1] = None                                    #Last element = None
		predicted = algorithm(train_set, test_set, *args)       #Use algorithm to return the array with the predicted values (Predictions set)
		actual = [row[-1] for row in fold]                      #Copy last element of each row of each fold to create (Observations set)
		accuracy = accuracy_metric(actual, predicted)           #Use accuracy_metric to get back the accuracy percentage of the tree (Score of 1 iteration)
		scores.append(accuracy)                                 #Append results to the scores list
	return scores                                            #Return the list with the scores 
 
def test_split(index, value, dataset):  #Split a dataset based on an attribute and an attribute value
	left, right = list(), list()         #Initialize two lists for left and right nodes
	for row in dataset:                    #For each row in dataset
		if row[index] < value:              #If the value of the row is less than the threshhold value
			left.append(row)                 #Append it to the left node
		else:                               #Else
			right.append(row)                #Append it to the right node
	return left, right                     #Return the 2 nodes
 
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	n_instances = float(sum([len(group) for group in groups])) #Count all samples at split point
	gini = 0.0                                                 #Sum weighted Gini index for each group
	for group in groups:                                         #For each group
		size = float(len(group))                                  #Get its size
		if size == 0:                                             # avoid divide by zero
			continue
		score = 0.0                                               #Initialize score
		for class_val in classes:                                 #For the values of each class
			p = [row[-1] for row in group].count(class_val) / size #Count all the last elements of each row of each group and divide with its size
			score += p * p                                         #score = score + p squared
		gini += (1.0 - score) * (size / n_instances)              #Weight the group score by its relative size
	return gini                                                #Return the evaluation
 
# Select the best split point for a dataset
def get_split(dataset, n_features):                             #Split the dataset
	class_values = list(set(row[-1] for row in dataset))         #Get last element of each row of the dataset (observations)
	b_index, b_value, b_score, b_groups = 999, 999, 999, None    #Initialize indices, score and group
	features = list()                                            #Create a new set of features
	while len(features) < n_features:                              #While no of features is less than the specified number
		index = randrange(len(dataset[0])-1)                        #Get a random index number out of the dataset
		if index not in features:                                   #If this index is not yet in features set
			features.append(index)                                   #Append it
	for index in features:                                       #For each feature
		for row in dataset:                                         #For each row of this feature
			groups = test_split(index, row[index], dataset)          #Decide which point go left or right in the dataset and add the arrays to groups
			gini = gini_index(groups, class_values)#Get the gini_index of all groups
			if gini < b_score:                                                       #If the gini index is greater than the old then 
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups #Replace old values
	return {'index':b_index, 'value':b_value, 'groups':b_groups}                   #Return the new values
 
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']                                           #Copy the two groups to left and right
	del(node['groups'])
	if not left or not right:                                              #Check for a no split
		node['left'] = node['right'] = to_terminal(left + right)            #If it didnt split then it is terminal
		return
	if depth >= max_depth:                                                 #Check for max depth
		node['left'], node['right'] = to_terminal(left), to_terminal(right) #If it reached max depth then it is terminal
		return
	if len(left) <= min_size:                                              #Process left child
		node['left'] = to_terminal(left)                                    #If it reached the minimun acceptable number of elements then it is terminal left leaf
	else:                                                                  #Else
		node['left'] = get_split(left, n_features)                          #Split the node in two again
		split(node['left'], max_depth, min_size, n_features, depth+1)       #Recurse increasing depth by 1
	if len(right) <= min_size:                                             #Process right child
		node['right'] = to_terminal(right)                                  #If it reached the minimun acceptable number of elements then it is terminal left leaf
	else:                                                                  #Else
		node['right'] = get_split(right, n_features)                        #Split the node in two again
		split(node['right'], max_depth, min_size, n_features, depth+1)      #Recurse increasing depth by 1
 
# Build a decision tree
def build_tree(train, max_depth, min_size, n_features): #Get dataset, (max depth), (min size)
	root = get_split(train, n_features)                  #Get the starting values for the dataset
	split(root, max_depth, min_size, n_features, 1)      #Split recursively until every tree in the forest is built
	return root
 
# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 
#Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):              
	sample = list()                         #Create an empty sample set
	n_sample = round(len(dataset) * ratio)  #Number of samples i will get from the dataset
	while len(sample) < n_sample:           #While samples requirements is not met
		index = randrange(len(dataset))      #Get the index of a random element from the dataset
		sample.append(dataset[index])        #Append it to samples set
	return sample                           #Return it
 
# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):                        
	predictions = [predict(tree, row) for tree in trees]           #For each tree give predictions
	return max(set(predictions), key=predictions.count)            #Get the best prediction possible out of them
 
# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()                                                 #Create a list of the trees
	for i in range(n_trees):                                       #For each tree
		sample = subsample(train, sample_size)                      #Get a random sample from the dataset
		tree = build_tree(sample, max_depth, min_size, n_features)  #Build the tree based on it
		trees.append(tree)                                          #Add the tree to your forest
	predictions = [bagging_predict(trees, row) for row in test]    #Predict the outcome of all trees
	return(predictions)                                            #Return Predictions
 
# Test the random forest algorithm
seed(2)                                                                               #Seed to always get same predictions
filename = '00.sonar.all-data.csv'                                                    #Name of the dataset
dataset = load_csv(filename)                                                          #Load dataset
for i in range(0, len(dataset[0])-1):                                                 #For each column of the dataset
	str_column_to_float(dataset, i)                                                    #If it is string -> convert to float
str_column_to_int(dataset, len(dataset[0])-1)                                         #Convert string lolumns to integers
n_folds = 2                                                                           #How many folds will we create out of the dataset
max_depth = 10                                                                        #Maximun depth the tree will reach
min_size = 1                                                                          #Minimun number of elements in each leaf
sample_size = 1.0                                                                     #How many samples will we create to test the forest
n_features = int(sqrt(len(dataset[0])-1))                                             #How many features will each tree hold
for n_trees in [1, 5, 10]:                                                            #Number of trees we will create
	scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)#Get the accuracy scores for each iteration of the tree
	print('Trees: %d' % n_trees)                                                       #Print the number of trees
	print('Scores: %s' % scores)                                                       #Print the scores
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))                  #Print their mean