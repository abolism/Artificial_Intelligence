"""
say we have a graph G with n nodes (0,1,...,n-1) and m edges
on every node there is a unique label from 0 to n-1
on every step we can swap the labels of the node with value of 0 with one of its neighbors
        we want to find the minimum number of steps to reach a state where all the nodes have the same label as their number (0,1,...,n-1)

we can use a BFS to find the minimum number of steps
we can use a set to store the states we have already visited
we can use a queue to store the states we have not yet visited
 in the first line of input we'll get n and m
 in the next m lines we'll get the edges of the graph
 in the last line we'll get the initial state of the graph
 for output we must print the minimum number of steps to reach the goal state
"""
from collections import deque
n, m = map(int, input().split())
graph = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)
state = list(map(int, input().split()))
visited = set()
queue = deque()
queue.append(state)
visited.add(tuple(state))
steps = 0
while queue:
    for _ in range(len(queue)):
        current_state = queue.popleft()
        if current_state == list(range(n)):
            print(steps)
            exit()
        for i in graph[current_state.index(0)]:
            new_state = current_state[:]
            new_state[current_state.index(0)] = new_state[i]
            new_state[i] = 0
            if tuple(new_state) not in visited:
                visited.add(tuple(new_state))
                queue.append(new_state)
    steps += 1
print(-1)

