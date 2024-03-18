import sys, time
from math import pi, acos, sin, cos
from time import perf_counter as pf
from heapq import heappop, heappush
import tkinter as tk

with open("rrEdges.txt") as f:
    edges = [line.strip() for line in f]

with open("rrNodes.txt") as f:
    nodes = {line.split()[0]: (float(line.split()[1]), float(line.split()[2])) for line in f}

with open("rrNodeCity.txt") as f:
    cities = {line.strip().split(' ', 1)[1]: line.strip().split(' ', 1)[0] for line in f}

# with open("north_america_boundaries.txt") as f:
#     boundaries = [tuple(map(float, line.strip().split())) for line in f]


train_routes = dict()


def calcd(node1, node2):
    if node1 == node2: return 0
    # y1 = lat1, x1 = long1
    # y2 = lat2, x2 = long2
    # all assumed to be in decimal degrees
    y1, x1 = node1
    y2, x2 = node2

    R = 3958.76  # miles = 6371 km
    y1 *= pi / 180.0
    x1 *= pi / 180.0
    y2 *= pi / 180.0
    x2 *= pi / 180.0

    # approximate great circle distance with law of cosines
    return acos(sin(y1) * sin(y2) + cos(y1) * cos(y2) * cos(x2 - x1)) * R


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
    path = {start: ''}
    while fringe:
        depth, cur = heappop(fringe)
        if cur == end:
            return depth, path
        if cur not in closed:
            closed.add(cur)
            for c in train_routes[cur]:
                if c[1] not in closed:
                    heappush(fringe, (depth + c[0], c[1]))
                    path[c[1]] = cur
    return None


def draw_dijkstra(start, end, canvas):
    if start == end:
        return 0
    closed = dict()
    fringe = [(0, start, ('', None))]
    n = 0
    while fringe:
        if n % 1000 == 0:
            canvas.update()
        depth, cur, par_edge = heappop(fringe)
        if cur == end:
            break
        if cur not in closed:
            closed[cur] = par_edge
            for c in train_routes[cur]:
                if c[1] not in closed:
                    temp = canvas.create_line(get_coords(nodes[cur]), get_coords(nodes[c[1]]), fill='red')
                    heappush(fringe, (depth + c[0], c[1], (cur, temp)))
        n += 1
    canvas.itemconfig(par_edge[1], width=1.5, fill='blue')
    par = par_edge[0]
    path = {end}
    while par != start:
        path.add(closed[par][1])
        par = closed[par][0]
    for edge in path:
        canvas.itemconfig(edge, width=1.5, fill='blue')
    canvas.update()
    return None


def astar(start, end):
    if start == end:
        return 0
    closed = set()
    fval = calcd(nodes[start], nodes[end])
    fringe = [(fval, 0, start)]
    path = {start: ''}
    while fringe:
        fval, depth, cur = heappop(fringe)
        if cur == end:
            return depth, path
        if cur not in closed:
            closed.add(cur)
            for c in train_routes[cur]:
                if c[1] not in closed:
                    fval = depth + c[0] + calcd(nodes[c[1]], nodes[end])
                    heappush(fringe, (fval, depth + c[0], c[1]))
                    path[c[1]] = cur
    return None


def draw_astar(start, end, canvas):
    if start == end:
        return 0
    closed = dict()
    fval = calcd(nodes[start], nodes[end])
    fringe = [(fval, 0, start, ('', None))]
    n = 0
    while fringe:
        if n % 100 == 0:
            canvas.update()
        fval, depth, cur, par_edge = heappop(fringe)
        if cur == end:
            break
        if cur not in closed:
            closed[cur] = par_edge
            for c in train_routes[cur]:
                if c[1] not in closed:
                    fval = depth + c[0] + calcd(nodes[c[1]], nodes[end])
                    temp = canvas.create_line(get_coords(nodes[c[1]]), get_coords(nodes[cur]), width=1, fill='red')
                    heappush(fringe, (fval, depth + c[0], c[1], (cur, temp)))
        n += 1

    canvas.itemconfig(par_edge[1], width=1.5, fill='blue')
    par = par_edge[0]
    path = {end}
    while par != start:
        path.add(closed[par][1])
        par = closed[par][0]
    for edge in path:
        canvas.itemconfig(edge, width=1.5, fill='blue')
    canvas.update()
    return None


def get_coords(node):
    lat, long = node
    a = (long + 135) * 14
    b = 675 - ((lat - 14) * 14)
    return a, b


def draw_map(canvas):
    routes = dict()
    for start, ends in train_routes.items():
        start_coords = get_coords(nodes[start])
        for end in ends:
            end_coord = get_coords(nodes[end[1]])
            route = canvas.create_line(start_coords, end_coord, width=1.15, fill='black')
            routes[start + end[1]] = route
    canvas.update()
    return routes


def main():
    global train_routes
    start, end ="Phoenix", "Atlanta" #sys.argv[1], sys.argv[2]
    startID, endID = cities[start], cities[end]
    train_routes = make_graph(edges, nodes)

    d_root = tk.Tk()
    d_root.geometry('1200x675+10+10')
    d_root.title('Train Routes')
    canvas = tk.Canvas(d_root, width=1200, height=675, bg='white')
    canvas.pack(anchor=tk.CENTER, expand=True)
    draw_map(canvas)
    canvas.create_text(1000, 575, text='Dijkstra', font=('Helvetica', '30', 'bold'), fill='red')
    draw_dijkstra(startID, endID, canvas)
    d_root.mainloop()

    a_root = tk.Tk()
    a_root.geometry('1200x675+10+10')
    a_root.title('Train Routes')
    canvas = tk.Canvas(a_root, width=1200, height=675, bg='white')
    canvas.pack(anchor=tk.CENTER, expand=True)
    draw_map(canvas)
    canvas.create_text(1000, 575, text='A*', font=('Helvetica', '30', 'bold'), fill='red')
    draw_astar(startID, endID, canvas)
    a_root.mainloop()


if __name__ == "__main__":
    main()
