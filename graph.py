import numpy as np
from collections import defaultdict

class Graph:
    # Create a graph
    # Input: None
    # Output: A graph
    # return np.zeros((10, 10))
    def __init__(self):
        self.adj_list = defaultdict(list)

    def add_edge(self, u, v, w):
        self.adj_list[u].append((v, w, "+"))
        self.adj_list[v].append((u, 1 / w, "-"))

    def get_path_between_u_and_v(self, u, v):
        global visited, result
        visited = set()

        def dfs(visited, node, path=1): 
            global kkk
            if node == v:
                kkk = path

            if node not in visited:
                visited.add(node)
                for neighbour, weight, sign in self.adj_list[node]:
                    dfs(visited, neighbour, path * weight)

        dfs(visited, u)
        return kkk
        print(kkk)

graph = Graph()

graph.add_edge("A", "B", 2)
graph.add_edge("A", "C", 8)
graph.add_edge("B", "C", 4)
graph.add_edge("B", "D", 8)
graph.add_edge("C", "D", 2)
graph.add_edge("C", "E", 7)

print(graph.get_path_between_u_and_v("C", "A"))