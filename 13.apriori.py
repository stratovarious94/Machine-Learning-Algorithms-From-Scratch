def Apriori_gen(Itemset, length):        #Get list with pruned values and the new length of the list
    canditate = []                       #List used to hold the combinations of each element
    for i in range (0,length):           #For each element in the dataset
        element = str(Itemset[i])          #Get is as string value
        for j in range (i+1,length):       #For each time it appears
            element1 = str(Itemset[j])       #Get it as string value
            if element[0:(len(element)-1)] == element1[0:(len(element1)-1)]:  #If they have the same names
                    unionset = element[0:(len(element)-1)]+element1[len(element1)-1]+element[len(element)-1] #Combine (k-1)-Itemset to k-Itemset 
                    unionset = ''.join(sorted(unionset))  #Sort itemset by dict order
                    canditate.append(unionset)  #Append the new combination set for the value
    return canditate   #Return the set with each combination

def Apriori_prune(Ck,MinSupport):   #Get list of elements and minimum number of times an element appears
    L = []                          #L is used to contain elements that appear an adequate number of times
    for i in Ck:                    #For each item in the list
        if Ck[i] >= minsupport:       #If it appears more times than the specified threshold
            L.append(i)                 #Append it to the list with the valid elements
    return sorted(L)                #Sort and return the list

def Apriori_count_subset(Canditate,Canditate_len):    #Get the list of the candidate subsets and its length
    Lk = dict()                                       #Create a new dictionary
    file = open('00.example.txt')                     #Open the file with the dataset again
    for l in file:                                    #For each line
        l = str(l.split())                              #Get an array with the row's elements as strings
        for i in range (0,Canditate_len):                 #For each candidate subset
            key = str(Canditate[i])                         #Get its value as string
            if key not in Lk:                               #If it is not registered in the dictionary yet
                Lk[key] = 0                                   #Include it as a key with 0 value
            flag = True                                     #Initialize a flag at True to use bellow
            for k in key:                                   #For each item of the subset
                if k not in l:                                #If it doesnt exist in file
                    flag = False                                #flag is false
            if flag:                                        #If the all the elements of the subset exist in the file's row
                Lk[key] += 1                                  #Increase the subset's frequency by 1
    file.close()                                      #Close the file
    return Lk                                         #Print how many times each subset appears in the file

minsupport = 3                  #If a rule exists less than 3 times it is excluded from the rules list
C1={}                           #C1 is a set to count how many each element appears in the dataset
file = open('00.example.txt')   #Get data from file
for line in file:               #Read them line by line
    for item in line.split():     #Read them element by element
        if item in C1:              #If the element is in C1
            C1[item] +=1              #Then increase its counter by one
        else:                       #Else
            C1[item] = 1              #Initialize it at 1
file.close()                       #Close the file
list(C1.keys()).sort()             #Sort by index key
L = []                             #L is used to contain elements that appear an adequate number of times
L1 = Apriori_prune(C1,minsupport)  #Returns a list of values which appear more than the threshold value
L = Apriori_gen(L1,len(L1))        #Combines those values to create a new set out of their combinations
print('====================================')  
print('Frequent 1-itemset is',L1)                   #Print the set with the items that pass the minsupport threshold
print('====================================')
k=2                                                 #Combinations complexity starts at 2
while L != []:                                      #As long as the list is not empty
    C = dict()                                      #Create a new dictionary
    C = Apriori_count_subset(L,len(L))              #Get how many times each subset appears
    frequent_itemset = []                           #Create an empty list to contain each of the sets that appear frequently
    frequent_itemset = Apriori_prune(C,minsupport)  #Filter them again by the minsupport threshold
    print('====================================')
    print('Frequent',k,'-itemset is',frequent_itemset) #Print the remaining subsets
    print('====================================')
    L = Apriori_gen(frequent_itemset,len(frequent_itemset))  #Generate new more complicated pairs until convergeance
    k += 1                                                   #Increase complexity by 1