import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_dataset(name):        #name = the name of the dataset
    return np.loadtxt(name)    #Load dataset in readable form and return it

def euclidian(a, b):
    return np.linalg.norm(a - b)

def kmeans(k, epsilon = 0, distance = 'euclidian'):  #k = number of clusters we want, epsilon = minimum error to stop the algorithm, distance = metric used to calculate the distance
    history_centroids = []                                             #List holding the centroid of every iteration
    if distance == 'euclidian':                                        #Check if distance method specified is 'euclidian'
        dist_method = euclidian                                        #dist_method = euclidian function
    dataset = load_dataset('00.durudataset.txt')                            #Load the dataset
    num_instances, num_features = dataset.shape                        #num_instances = number of rows of the dataset, num_features = number of columns of the dataset
    prototypes = dataset[np.random.randint(0, num_instances - 1, size = k)]  #Get the indices of k random rows of the dataset to use as centroids
    history_centroids.append(prototypes)                               #Append the centroids to the history list so that we can later visualize the algorithms choices
    prototypes_old = np.zeros(prototypes.shape)                        #Used to hold the centroids of the previous iteration
    belongs_to = np.zeros((num_instances, 1))                          #Create a vector which will hold the smaller distance between each row and the centroids
    norm = dist_method(prototypes, prototypes_old)                     #Calculate error between the current centroids and the old ones
    iterations = 0                                                     #Initialize iterations counter
    while norm > epsilon:                                              #While the error is greater than our threshold (0)
        iterations += 1                                                  #Increase the iterations counter by 1
        norm = dist_method(prototypes, prototypes_old)                   #Calculate error between the centroids
        for index_instance, instance in enumerate(dataset):              #For each row in the dataset get its index and value
            dist_vec = np.zeros((k, 1))                                    #Create a distance vector of size k to hold the distance between a row and each centroid
            for index_prototype, prototype in enumerate(prototypes):       #For each centroid get its index and value
                dist_vec[index_prototype] = dist_method(prototype, instance) #Calculate the distances between a row and the k centroids and store them in the vector
            belongs_to[index_instance, 0] = np.argmin(dist_vec)            #belongs_to(index of the row of respective instance) = index of centroid with the smallest value
        tmp_prototypes = np.zeros((k, num_features))                     #Buffer holding the centroids
        for index in range(len(prototypes)):                                                #For each centroid
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]   #Store all points by cluster in instances_close
            prototype = np.mean(dataset[instances_close], axis=0)                             #Create the new centroid by calculating the means of each cluster
            tmp_prototypes[index, :] = prototype                                              #Store the centroid to the tmp list
        prototypes = tmp_prototypes                                                         #Copy the new list to the prototypes list
        history_centroids.append(tmp_prototypes)                                            #Append the new centroids to history
    return prototypes, history_centroids, belongs_to                                      #Upon convergeance return the last centroids, their history and where each point belongs to

def plot(dataset, history_centroids, belongs_to):
  colors = ['r', 'g']
  fig, ax = plt.subplots()
  for index in range(dataset.shape[0]):
    instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
    for instance_index in instances_close:
      ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))
      history_points = []
    for index, centroids in enumerate(history_centroids):
      for inner, item in enumerate(centroids):
        if index == 0:
          history_points.append(ax.plot(item[0], item[1], 'bo')[0])
        else:
          history_points[inner].set_data(item[0], item[1])
          print("centroids {} {}".format(index, item))
          plt.pause(0.8)

def execute():
    dataset = load_dataset('00.durudataset.txt')
    centroids, history_centroids, belongs_to = kmeans(2)
    plot(dataset, history_centroids, belongs_to)

execute()
            
            
        