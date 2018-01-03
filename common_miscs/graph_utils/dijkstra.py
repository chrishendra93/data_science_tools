from collections import defaultdict
from heapq import *


def dijkstra(edges, f, t):
    g = defaultdict(list)
    for c, l, r in edges:
        g[l].append((c, r))

    q, seen = [(0, f, [])], set()
    while q:
        (cost, v1, path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = [v1] + path
            if v1 == t:
                return (cost, [(path[i], path[i - 1]) for i in range(len(path) - 1, 0, -1)])

            for c, v2 in g.get(v1, ()):
                if v2 not in seen:
                    heappush(q, (cost + c, v2, path))


if __name__ == "__main__":

    edges = [
        (7, "A", "B"),
        (5, "A", "D"),
        (1, "B", "C"),
        (9, "B", "D"),
        (7, "B", "E"),
        (5, "C", "E"),
        (15, "D", "E"),
        (6, "D", "F"),
        (1, "F", "E"),
        (9, "E", "G"),
        (11, "F", "G")
    ]
    print edges
    print "=== Dijkstra ==="
    print edges
    print "A -> E:"
    print dijkstra(edges, "A", "E")
    print "F -> G:"
    print dijkstra(edges, "F", "G")
