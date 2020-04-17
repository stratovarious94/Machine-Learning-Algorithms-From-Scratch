import numpy as np
import pylab as plt
import networkx as nx           #Used to create a graph

# map cell to cell, add circular cell to goal point
points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]

goal = 7

G=nx.Graph()                    #Initialize a new graph as G
G.add_edges_from(points_list)   #Feed graph with the edges
pos = nx.spring_layout(G)       #Generate their positions
nx.draw_networkx_nodes(G,pos)   #Draw nodes
nx.draw_networkx_edges(G,pos)   #Draw edges
nx.draw_networkx_labels(G,pos)  #Insert labels in each node
plt.show()                      #Draw

# how many points in graph? x points
MATRIX_SIZE = 8  #N+1 routes

# create matrix x*y
R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))  #Initialize matrix
R *= -1                                                   #Make all elements -1

# assign zeros to paths and 100 to goal-reaching point
for point in points_list:        #For each point
    print(point)                 #Print it
    if point[1] == goal:           #If the destination is the goal
        R[point] = 100               #Give it 100 points
    else:                          #Else
        R[point] = 0                 #It is another room so it gets 0 points
    #For going backwards
    if point[0] == goal:           #If the source
        R[point[::-1]] = 100         #Is the goal give it 100
    else:                          #Else
        R[point[::-1]]= 0            #It is another room so it gets 0 points

# add goal point round trip
R[goal,goal]= 100                  #Last element has to be the goal

print(R)                           #Print resulting map

Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))

# learning parameter
gamma = 0.8                    #How fast will it converge
initial_state = 1              #The starting node is 1

def available_actions(state):     #Gets current node
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act                 #Returns all availiable routes from there

available_act = available_actions(initial_state) 

def sample_next_action(available_actions_range):  #Gets availiable actions
    next_action = int(np.random.choice(available_act,1))
    return next_action                            #Chooses a new action randomly

action = sample_next_action(available_act)  #Generate next action

def update(current_state, action, gamma):  #Gets the current node, the action it took and the learning rate
    
  max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
  
  if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
  else:
      max_index = int(max_index)
  max_value = Q[action, max_index]
  
  Q[current_state, action] = R[current_state, action] + gamma * max_value
  print('max_value', R[current_state, action] + gamma * max_value)
  
  if (np.max(Q) > 0):
    return(np.sum(Q/np.max(Q)*100))    #Registers 100 if it took the best action
  else:
    return (0)                         #Or 0 if it did not
    
update(initial_state, action, gamma)   #Play the first game
# Training
scores = []                            #Board mapping the rewards for each action of the bot
for i in range(700):                                       #Play the game 700 times
    current_state = np.random.randint(0, int(Q.shape[0]))  #Start at a random node
    available_act = available_actions(current_state)       #Get the availiable actions
    action = sample_next_action(available_act)             #Choose one of them according to the reward data until now
    score = update(current_state,action,gamma)             #Play a game and get the final score
    scores.append(score)                                   #Append it to scores list to draw them later
    print ('Score:', str(score))                           #Print it
    
print("Trained Q matrix:")
print(Q/np.max(Q)*100)     #Print best score

# Testing
current_state = 0          #Test the results
steps = [current_state]

while current_state != 7:  #While the current state is not the goal play the optimal path it found when training

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)     #Print the path

plt.plot(scores) #Plot the algorithm as it converges to optimal
plt.show()