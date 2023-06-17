#
#

# import numpy as np
# n, m, z = list(map(int, input().split()))
# adj = np.zeros((n, n))
# observed = np.zeros(n)
# descendants = [None] * n
# for _ in range(m):
#     x, y = list(map(int, input().split()))
#     adj[x - 1][y - 1] = 1
# for _ in range(z):
#     x = int(input())
#     observed[x - 1] = 1
# def calc_desc(node):
#     res = [node]
#     for i in range(n):
#         if adj[node][i] == 1:
#             calc_desc(i)
#             for item in descendants[i]:
#                 if item not in res:
#                     res.append(item)
#     descendants[node] = res
# def is_node_or_descs_observed(node):
#     for d in descendants[node]:
#         if observed[d] == 1:
#             return True
#     return False
# for i in range(n):
#     if descendants[i] is None:
#         calc_desc(i)
# path = []
# all_paths = []
# def get_all_paths(source, dest):
#     for i in range(n):
#         if adj[i][source] == 1 or adj[source][i] == 1:
#             if i == dest:
#                 path.append(i)
#                 all_paths.append(path.copy())
#                 path.pop()
#             elif i not in path:
#                 path.append(i)
#                 get_all_paths(i, dest)
#                 path.pop()
# source, dest = list(map(int, input().split()))
# source -= 1
# dest -= 1
# path.append(source)
# get_all_paths(source, dest)
# ans = False
# def is_blocked(path):
#     for i in range(len(path) - 2):
#         u, v, w = path[i], path[i + 1], path[i + 2]
#         if adj[u][v] == 1 and adj[v][w] == 1 and observed[v] == 1:
#             return True
#         if adj[v][u] == 1 and adj[w][v] == 1 and observed[v] == 1:
#             return True
#         if adj[v][u] == 1 and adj[v][w] == 1 and observed[v] == 1:
#             return True
#         if adj[u][v] == 1 and adj[w][v] == 1 and not is_node_or_descs_observed(v):
#             return True
#     return False
# for p in all_paths:
#     if not is_blocked(p):
#         ans = True
#         str_p = [str(x + 1) for x in p]
#         print(', '.join(str_p))
# if not ans:
#     print("independent")




from collections import defaultdict
class Graph:
    def getAllTriplets(self, path):
        triplets = []
        for i in range(len(path) - 2):
            triplets.append([path[i], path[i + 1], path[i + 2]])
        return triplets

    def allActiveForms(self, triplet):
        directions = [0, 0]
        if self.graph[triplet[0]].count(triplet[1]) > 0:
            directions[0] = 1
        if self.graph[triplet[1]].count(triplet[2]) > 0:
            directions[1] = 1

        if directions[0] == 1 and directions[1] == 0:
            if triplet[1] in Z or self.childrenInZ(triplet[1], Z):
                return True
            return False
        if triplet[1] not in Z:
            return True
        return False




    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices

        # default dictionary to store graph
        self.graph = defaultdict(list)
        self.undirected_graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u-1].append(v-1)
        self.undirected_graph[v-1].append(u-1)
        self.undirected_graph[u-1].append(v-1)

    def childrenInZ(self, i, Z):
        for j in self.graph[i]:
            if j in Z:
                return True
            if self.childrenInZ(j,Z):
                return True
        return False

    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''

    def printAllPathsUtil(self, u, d, visited, path):

        # Mark the current node as visited and store in path
        visited[u] = True
        path.append(u)

        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            allTriplets = self.getAllTriplets(path)
            for triplet in allTriplets:
                if not self.allActiveForms(triplet):
                    return
            print(', '.join(str(i + 1) for i in path))
            # print(', '.join(path))
            exit()



            # is_active = False
            # if len(path) == 3:
            #     triplet = path
            #     # now we check if the triplet is active or not
            #     directions = [0, 0]
            #     if self.graph[triplet[0]].count(triplet[1]) > 0:
            #         directions[0] = 1
            #     if self.graph[triplet[1]].count(triplet[2]) > 0:
            #         directions[1] = 1
            #
            #     if directions[0] == 1 and directions[1] == 1:
            #         if triplet[1] in Z:
            #             is_active = False
            #             # break
            #         else:
            #             # if i == len(path) - 3:
            #             is_active = True
            #             print(",".join(str(i + 1) for i in path))
            #             exit()
            #     if directions[0] == 1 and directions[1] == 0:
            #         if triplet[1] not in Z and not self.childrenInZ(triplet[1], Z):
            #             is_active = False
            #             # break
            #         else:
            #             # if i == len(path) - 3:
            #             is_active = True
            #             print(",".join(str(i + 1) for i in path))
            #             exit()
            #
            #     if directions[0] == 0 and directions[1] == 1:
            #         if triplet[1] in Z:
            #             is_active = False
            #             # break
            #         else:
            #             # if i == len(path) - 3:
            #             is_active = True
            #             print(",".join(str(i + 1) for i in path))
            #             exit()
            #
            #     if directions[0] == 0 and directions[1] == 0:
            #         if triplet[1] in Z:
            #             is_active = False
            #             # break
            #         else:
            #             # if i == len(path) - 3:
            #             is_active = True
            #             print(",".join(str(i + 1) for i in path))
            #             exit()
            #
            # if len(path)>3:
            #     for i in range(len(path)-2):
            #         triplet = [path[i],path[i+1],path[i+2]]
            #         #now we check if the triplet is active or not
            #         directions = [0,0]
            #         if self.graph[triplet[0]].count(triplet[1]) > 0:
            #             directions[0] = 1
            #         if self.graph[triplet[1]].count(triplet[2]) > 0:
            #             directions[1] = 1
            #
            #         if directions[0] == 1 and directions[1] == 1:
            #             if triplet[1] in Z:
            #                 is_active = False
            #                 break
            #             else:
            #                 if i == len(path)-3:
            #                     is_active = True
            #                     print(",".join(str(i + 1) for i in path))
            #                     exit()
            #
            #         if directions[0] == 1 and directions[1] == 0:
            #             if triplet[1] not in Z and not self.childrenInZ(triplet[1],Z):
            #                 is_active = False
            #                 break
            #             else:
            #                 if i == len(path)-3:
            #                     is_active = True
            #                     print(",".join(str(i + 1) for i in path))
            #                     exit()
            #
            #         if directions[0] == 0 and directions[1] == 1:
            #             if triplet[1] in Z:
            #                 is_active = False
            #                 break
            #             else:
            #                 if i == len(path)-3:
            #                     is_active = True
            #                     print(",".join(str(i + 1) for i in path))
            #                     exit()
            #
            #         if directions[0] == 0 and directions[1] == 0:
            #             if triplet[1] in Z:
            #                 is_active = False
            #                 break
            #             else:
            #                 if i == len(path)-3:
            #                     is_active = True
            #                     print(",".join(str(i + 1) for i in path))
            #                     exit()





        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.undirected_graph[u]:
                if visited[i] == False:
                    # print(i, "i")
                    self.printAllPathsUtil(i, d, visited, path)

        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u] = False

    # Prints all paths from 's' to 'd'
    def printAllPaths(self, s, d):

        # Mark all the vertices as not visited
        visited = [False] * (self.V)

        # Create an array to store paths
        path = []

        # Call the recursive helper function to print all paths
        self.printAllPathsUtil(s, d, visited, path)
        print("independent")

    def find_decandents(self, i,decandents = []):
        cntr = 0
        for j in self.undirected_graph[i]:
            if j not in decandents and j not in self.graph[i]:
                decandents.append(j)
                self.find_decandents(j,decandents)
        if cntr == 0:
            return decandents


# Create a graph given in the above diagram
n,m,z = [int(i) for i in input().split()]
g = Graph(n)
for i in range(m):
    a, b = map(int, input().split())
    g.addEdge(a, b)
#now we input the set Z
Z = []
for i in range(z):
    Z.append(int(input()) - 1)
#now we input the 2 nodes that we want to check if they are independent or not
a, b = map(int, input().split())
#now we check if they are independent or not
decendants = []
for i in Z:
    decandents = g.find_decandents(i)
    decendants.extend(decandents)

# print(decendants)
g.printAllPaths(a-1, b-1)
#we want to add every parent and grand parent and grand grand parent and so on of every member of Z to a list

