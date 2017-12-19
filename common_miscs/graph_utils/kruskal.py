class Kruskal:

    def __init__(self, graph, mode="max"):
        self.parent = dict()
        self.rank = dict()
        self.graph = graph
        self.mode = mode

    def make_set(self, vertex):
        self.parent[vertex] = vertex
        self.rank[vertex] = 0

    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]

    def union(self, vertex1, vertex2):
        root1 = self.find(vertex1)
        root2 = self.find(vertex2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1

    def get_mst(self, mode='minimum'):
        '''this method can be used to obtain minimum or maximum spanning tree'''
        '''multiplier variable below will reverse the sign of the edge weight of the graph'''
        multiplier = 1 if mode == 'minimum' else -1
        for vertex in self.graph['vertices']:
            self.make_set(vertex)
        minimum_spanning_tree = set()
        edges = self.graph['edges']
        if self.mode == "max":
            edges = list({(edge[0] * multiplier, edge[1], edge[2]) for edge in self.graph['edges']})
        elif self.mode == "min":
            edges = list(graph['edges'])
        else:
            raise ValueError("Invalid mode")
        edges.sort()
        for edge in edges:
            weight, vertex1, vertex2 = edge
            if self.find(vertex1) != self.find(vertex2):
                self.union(vertex1, vertex2)
                minimum_spanning_tree.add(edge)
        return minimum_spanning_tree


if __name__ == '__main__':
    graph = {
            'vertices': ['A', 'B', 'C', 'D', 'E', 'F'],
            'edges': set([
                (1, 'A', 'B'),
                (5, 'A', 'C'),
                (3, 'A', 'D'),
                (4, 'B', 'C'),
                (2, 'B', 'D'),
                (1, 'C', 'D'),
                ])
            }
    minimum_spanning_tree = set([
                (1, 'A', 'B'),
                (2, 'B', 'D'),
                (1, 'C', 'D'),
                ])
    kruskal = Kruskal(graph)
    assert minimum_spanning_tree == kruskal.get_mst()

    print kruskal.get_mst()
