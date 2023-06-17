import numpy as np


k = int(input())
n = int(input())
m = int(input())

'''
we have n possible states (Xi a member of {1,2,...,n})
we have k observations of what E has been
we have m possible emissions (E a member of {1,2,...,m})
we have a transition matrix of size n x n
we have an emission matrix of size n x m
we have an initial state distribution of size n x 1 (Probablity of X1 = i for i in {1,2,...,n})
we have an observation sequence of size k x 1 for emission E1, E2, ..., Ek
we must calculate the e in {1,2,...,m} that Ek+1 = e is the most likely and also the probability of Ek+1 = e
'''

#now we input the observation sequence
observation_sequence = [int(i) for i in input().split()]
observation_sequence = np.array(observation_sequence)

#now we input the initial state distribution
initial_state_distribution = [float(i) for i in input().split()]
initial_state_distribution = np.array(initial_state_distribution)

#we now input the transition matrix
transition_matrix = []
for i in range(n):
    transition_matrix.append([float(i) for i in input().split()])
transition_matrix = np.array(transition_matrix)

#we then input the emission matrix
emission_matrix = []
for i in range(n):
    emission_matrix.append([float(i) for i in input().split()])
emission_matrix = np.array(emission_matrix)

#now we define a function that takes two states x and y and returns the probability of transitioning from x to y
def transition_probability(x, y):
    return transition_matrix[x][y]

#now we define a function that takes two states x and e and returns the probability of emitting e from x
def emission_probability(x, e):
    return emission_matrix[x][e]





pX1 = initial_state_distribution
listOfP = [initial_state_distribution.tolist()]
# print(listOfP)
# print(observation_sequence)

for x in range(1,k):
    ekByxk = [1 for kdfj in range(n)]
    for i in range(n):
        pX = [0 for kk in range(n)]
        for j in range(n):
            # pXX = [0 for kdfsdjfk in range(n)]
            hold = listOfP[-1][j] * transition_probability(j, i)
            pX[i] += hold * emission_probability(j, observation_sequence[x]-2)
        ekByxk[i] = pX[i] * emission_probability(i, observation_sequence[x]-1)
    listOfP.append(ekByxk)
    # if x == 0:
    #     listOfP.pop(0)

# print(listOfP)

#now we must calculate xk+1 for every possible value of xk+1 in {1,2,...,n}
pXkk = [0 for i in range(n)]
for i in range(n):
    for j in range(n):
        pXkk[i] += listOfP[-1][j] * transition_probability(j, i)

listOfP.append(pXkk)

#now we must calculate the most likely e in {1,2,...,m} that Ek+1 = e and also the probability of Ek+1 = e using emm=ission_matrix
mostLikelyE = 0
mostLikelyEProb = 0
listOfE = []
for i in range(m):
    pForThisE = 0
    for j in range(n):
        pForThisE += pXkk[j] * emission_probability(j, i)
    listOfE.append(pForThisE)

# print(listOfP[-1] , "listOfP[-1]")
# print(emission_matrix, "emission_matrix")


# print(listOfE)
#now we want to get the sum of members of listOfE
sumOfE = 0
for i in range(len(listOfE)):
    sumOfE += listOfE[i]

for i in range(len(listOfE)):
    listOfE[i] /= sumOfE

# print(listOfE)

# print(listOfE)
#now we want to print the maximum value of listOfE and its index
for i in range(len(listOfE)):
    if listOfE[i] > mostLikelyEProb:
        mostLikelyEProb = listOfE[i]
        mostLikelyE = i

print(mostLikelyE+1, round(mostLikelyEProb, 2))

# print(emission_matrix)
# for i in range(n):
#     for j in range(m):
#         print(emission_probability(i, j))
#         if j == len(emission_matrix[i]) - 1:
#             print("")

# print(listOfP)
# print(transition_matrix)

# for i in range(n):
#     for j in range(n):
#         print(transition_probability(i, j))





