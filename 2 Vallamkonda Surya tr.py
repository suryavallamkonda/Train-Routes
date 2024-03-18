import sys, time
from time import perf_counter as pf
from math import pi , acos , sin , cos
from heapq import heappop, heappush

with open("rrEdges.txt") as f:
    edges = [line.strip() for line in f]

with open("rrNodes.txt") as f:
    nodes = {line.split()[0]: (float(line.split()[1]), float(line.split()[2])) for line in f}

with open("rrNodeCity.txt") as f:
    cities = {line.strip().split(' ', 1)[1]: line.strip().split(' ', 1)[0] for line in f}

train_routes = dict()

def calcd(node1, node2):
   if node1 == node2: return 0
   # y1 = lat1, x1 = long1
   # y2 = lat2, x2 = long2
   # all assumed to be in decimal degrees
   y1, x1 = node1
   y2, x2 = node2

   R   = 3958.76 # miles = 6371 km
   y1 *= pi/180.0
   x1 *= pi/180.0
   y2 *= pi/180.0
   x2 *= pi/180.0

   # approximate great circle distance with law of cosines
   return acos( sin(y1)*sin(y2) + cos(y1)*cos(y2)*cos(x2-x1) ) * R


def make_graph(edges, nodes):
    train_routes = dict()
    for edge in edges:
        c1, c2 = edge.split()
        if c1 in train_routes:
            train_routes[c1].append((calcd(nodes[c1], nodes[c2]), c2))
        else:
            train_routes[c1] = [(calcd(nodes[c1], nodes[c2]), c2)]
        if c2 in train_routes:
            train_routes[c2].append((calcd(nodes[c2], nodes[c1]), c1))
        else:
            train_routes[c2] = [(calcd(nodes[c2], nodes[c1]), c1)]
    return train_routes


def dijkstra(start, end):
    if start == end:
        return 0
    closed = set()
    fringe = [(0, start)]
    while fringe:
        depth, cur = heappop(fringe)
        if cur == end:
            return depth
        if cur not in closed:
            closed.add(cur)
            for c in train_routes[cur]:
                if c[1] not in closed:
                    heappush(fringe, (depth + c[0], c[1]))
    return None


def astar(start, end):
    if start == end:
        return 0
    closed = set()
    fval = calcd(nodes[start], nodes[end])
    fringe = [(fval, 0, start)]
    while fringe:
        fval, depth, cur = heappop(fringe)
        if cur == end:
            return depth
        if cur not in closed:
            closed.add(cur)
            for c in train_routes[cur]:
                if c[1] not in closed:
                    fval = depth + c[0] + calcd(nodes[c[1]], nodes[end])
                    heappush(fringe, (fval, depth + c[0], c[1]))
    return None

def main():
    global train_routes
    start, end = sys.argv[1], sys.argv[2]
    start1 = pf()
    train_routes = make_graph(edges, nodes)
    print(f"Time to create data structure: {pf() - start1}")
    startID, endID = cities[start], cities[end]
    dstart = pf()
    test = dijkstra(startID, endID)
    print(f"{start} to {end} with Dijkstra: {test} in {pf()-dstart} seconds.")
    astart = pf()
    test2 = astar(startID, endID)
    print(f"{start} to {end} with A*: {test2} in {pf()-astart} seconds.")


if __name__ == "__main__":
    main()