import sys, time
from collections import deque
from math import pi, acos, sin, cos
from time import perf_counter as pf
from heapq import heappop, heappush
from threading import *
import multiprocessing
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
            train_routes[c1].append((calcd(nodes[c1], nodes[c2]), c2)) # train_routes[vertex1].append((cost from vertex1 to vertex2, vertex2))
        else:
            train_routes[c1] = [(calcd(nodes[c1], nodes[c2]), c2)]
        if c2 in train_routes:
            train_routes[c2].append((calcd(nodes[c2], nodes[c1]), c1))
        else:
            train_routes[c2] = [(calcd(nodes[c2], nodes[c1]), c1)]
    return train_routes


def get_coords(node): # x, y from lat, long
    lat, long = node
    a = (long + 135) * 13
    b = 675 - ((lat - 10) * 13)
    return a, b


def draw_map():
    routes = dict()
    for start, ends in train_routes.items():
        start_coords = get_coords(nodes[start])
        for end in ends:
            end_coord = get_coords(nodes[end[1]])
            route = canvas.create_line(start_coords, end_coord, width=1.15, fill='black')
            routes[start + end[1]] = route
            routes[end[1] + start] = route
    return routes


def draw_dijkstra(start, end):
    if start == end:
        return 0
    closed = dict()
    colored = set()
    fringe = [(0, start, ('', None))]
    n = 0
    while fringe:
        if n % 1000 == 0:
            canvas.update()
        depth, cur, par_edge = heappop(fringe)
        depthLabel.config(text = f'Current distance: {depth}', font = ('Helvetica', '15'))
        if cur == end:
            break
        if cur not in closed:
            closed[cur] = par_edge
            for c in train_routes[cur]:
                if c[1] not in closed:
                    temp = routes[c[1] + cur]
                    canvas.itemconfig(temp, width=1.25, fill='red')
                    colored.add(temp)
                    heappush(fringe, (depth + c[0], c[1], (cur, temp)))
        n += 1
    canvas.itemconfig(par_edge[1], width=1.75, fill='blue')
    par = par_edge[0]
    path = {end}
    while par != start:
        path.add(closed[par][1])
        par = closed[par][0]
    for edge in path:
        canvas.itemconfig(edge, width=1.75, fill='blue')
    canvas.update()
    return colored


def draw_astar(start, end):
    if start == end:
        return 0
    closed = dict()
    colored = set()
    fval = 0
    for i in train_routes[start]: fval = i[0] if i[1] == end else None
    fringe = [(fval, 0, start, ('', None))]
    n = 0
    while fringe:
        if n % 100 == 0:
            canvas.update()
        fval, depth, cur, par_edge = heappop(fringe)
        depthLabel.config(text = f'Current distance: {depth}', font = ('Helvetica', '15'))
        if cur == end:
            break
        if cur not in closed:
            closed[cur] = par_edge
            for c in train_routes[cur]:
                if c[1] not in closed:
                    fval = depth + c[0] + calcd(nodes[c[1]], nodes[end])
                    temp = routes[cur + c[1]]
                    canvas.itemconfig(temp, width=1.25, fill='red')
                    colored.add(temp)
                    heappush(fringe, (fval, depth + c[0], c[1], (cur, temp)))
        n += 1
    canvas.itemconfig(par_edge[1], width=1.75, fill='blue')
    par = par_edge[0]
    path = {end}
    while par != start:
        path.add(closed[par][1])
        par = closed[par][0]
    for edge in path:
        canvas.itemconfig(edge, width=1.75, fill='blue')
    canvas.update()
    return colored


def draw_dfs(start, end):
    colored = set()
    fringe = deque()
    closed = {start: ('', None, None)}
    fringe.append((start, 0, ('', None)))
    n = 0
    while fringe:
        if n % 10 == 0: canvas.update()
        cur, depth, par_edge = fringe.pop()
        depthLabel.config(text = f'Current distance: {depth}', font = ('Helvetica', '15'))
        if cur == end:
            break
        for c in train_routes[cur]:
            if c[1] not in closed:
                temp = routes[cur + c[1]]
                canvas.itemconfig(temp, width=1.25, fill='red')
                colored.add(temp)
                fringe.append((c[1], depth + c[0], (cur, temp)))
                closed[c[1]] = (cur, temp, depth + c[0])
        n += 1
    canvas.itemconfig(par_edge[1], width=1.75, fill='blue')
    par = par_edge[0]
    path = {end}
    while par != start:
        path.add(closed[par][1])
        par = closed[par][0]
    for edge in path:
        canvas.itemconfig(edge, width=1.75, fill='blue')
    canvas.update()
    return colored


def kdfs(k, start, end):
    colored = set()
    ancestors = {start: start}
    fringe = [(start, 0, ancestors, 0)]
    n = 0
    while fringe:
        if n % (int((k ** 2)//2) +1) == 0: canvas.update()
        state, depth, ancestors, true_depth = fringe.pop()
        depthLabel.config(text = f'Current distance: {true_depth}', font = ('Helvetica', '15'))
        if state == end:
            break
        if depth > k: break
        if depth < k:
            for c in train_routes[state]:
                if c[1] not in ancestors:
                    temp = ancestors.copy()
                    temp[c[1]] = state
                    line = routes[state + c[1]]
                    canvas.itemconfig(line, width=1.25, fill='red')
                    colored.add(line)
                    fringe.append((c[1], depth + 1, temp, depth + c[0]))
        n += 1
    if state == end:
        path = {end}
        while state != start:
            path.add(ancestors[state])
            state = ancestors[state]
        for edge in path:
            canvas.itemconfig(edge, width=1.75, fill='blue')
        return 1, colored
    else:
        return None, colored


def draw_iddfs(start, end):
    max_depth = 0
    result, colored = None, set()
    while result is None:
        result, colored = kdfs(max_depth, start, end)
        max_depth += 1
        if result is None:
            for edge in colored: canvas.itemconfig(edge, width = 1, fill = 'black')
        canvas.update()
    return colored


def draw_bidijkstra(start, end):
    colored = set()
    intersections = set()
    start_fringe, end_fringe = [(0, start)], [(0, end)]
    start_closed, end_closed = {start: (0, None)}, {end: (0, None)} # current : (depth, parent)
    n = 0
    node = None
    while start_fringe and end_fringe:
        if n % 100 == 0: canvas.update()
        depth, node = heappop(start_fringe)
        if node in end_closed:
            depthLabel.config(text = f'Current distance from start: {depth}\nCurrent distance from end: {heappop(end_fringe)[0]}', font = ('Helvetica', '10'))
            break
        intersections.add(node)
        for c in train_routes[node]:
            if c[1] not in intersections:
                c_depth = depth + c[0]
                if c[1] in start_closed and c_depth < start_closed[c[1]][0] or c[1] not in start_closed:
                    heappush(start_fringe, (c_depth, c[1]))
                    start_closed[c[1]] = (c_depth, node)
                    edge = routes[c[1] + node]
                    canvas.itemconfig(edge, width = 1.25, fill = 'red')
                    colored.add(edge)
        s_depth = depth
        depth, node = heappop(end_fringe)
        depthLabel.config(text = f'Current distance from start: {s_depth}\nCurrent distance from end: {depth}', font = ('Helvetica', '10'))
        if node in start_closed:
            break
        intersections.add(node)
        for c in train_routes[node]:
            if c[1] not in intersections:
                c_depth = depth + c[0]
                if c[1] in end_closed and c_depth < end_closed[c[1]][0] or c[1] not in end_closed:
                    heappush(end_fringe, (c_depth, c[1]))
                    end_closed[c[1]] = (c_depth, node)
                    edge = routes[c[1] + node]
                    canvas.itemconfig(edge, width = 1.25, fill = 'red')
                    colored.add(edge)
        n += 1
    path = set()
    temp = node
    while temp != end:
        path.add(routes[end_closed[temp][1] + temp])
        temp = end_closed[temp][1]
    temp = node
    while temp != start:
        path.add(routes[start_closed[temp][1] + temp])
        temp = start_closed[temp][1]
    for edge in path:
        canvas.itemconfig(edge, width = 1.75, fill = 'blue')
    canvas.update()
    return colored


def draw_reverse_astar(start, end):
    if start == end:
        return 0
    closed = dict()
    colored = set()
    fval = calcd(nodes[start], nodes[end])
    fringe = [(fval, 0, start, ('', None))]
    n = 0
    while fringe:
        if n % 100 == 0:
            canvas.update()
        fval, depth, cur, par_edge = heappop(fringe)
        depthLabel.config(text = f'Current distance: {depth}', font = ('Helvetica', '15'))
        if cur == end:
            break
        if cur not in closed:
            closed[cur] = par_edge
            for c in train_routes[cur]:
                if c[1] not in closed:
                    fval = depth + c[0] + calcd(nodes[c[1]], nodes[end])
                    temp = routes[cur + c[1]]
                    canvas.itemconfig(temp, width=1.25, fill='red')
                    colored.add(temp)
                    heappush(fringe, (-1 * fval, depth + c[0], c[1], (cur, temp)))
        n += 1
    canvas.itemconfig(par_edge[1], width=1.75, fill='blue')
    par = par_edge[0]
    path = {end}
    while par != start:
        path.add(closed[par][1])
        par = closed[par][0]
    for edge in path:
        canvas.itemconfig(edge, width=1.75, fill='blue')
    canvas.update()
    return colored


def draw_temp(start, end):
    'e'


def display_selected():
    global temp_colored
    for route in temp_colored: canvas.itemconfig(route, width=1, fill='black')
    depthLabel.config(text = '')
    canvas.update()
    choice = default.get()
    if choice == 'Choose an \nalgorithm': return
    if choice == 'Dijkstra':
        colored = draw_dijkstra(startID, endID)
        temp_colored = colored
    elif choice == 'A*':
        colored = draw_astar(startID, endID)
        temp_colored = colored
    elif choice == 'DFS':
        colored = draw_dfs(startID, endID)
        temp_colored = colored
    elif choice == 'ID-DFS':
        colored = draw_iddfs(startID, endID)
        temp_colored = colored
    elif choice == 'Bidirectional Dijkstra':
        colored = draw_bidijkstra(startID, endID)
        temp_colored = colored
    elif choice == 'Reverse A*':
        colored = draw_reverse_astar(startID, endID)
        temp_colored = colored
    elif choice == 'temp':
        colored = draw_temp(startID, endID)
        temp_colored = colored


def stoprun():
    while True:
        'e'
        

temp_colored = set()


def main():
    global train_routes, canvas, startID, endID, default, routes, depthLabel
    start, end = 'Leon', 'Tucson'  # sys.argv[1], sys.argv[2]
    startID, endID = cities[start], cities[end]
    train_routes = make_graph(edges, nodes)

    root = tk.Tk()
    root.geometry('1350x775+0+0')
    root.title('Train Routes')
    canvas = tk.Canvas(root, width=1200, height=650, bg='white')
    canvas.pack(anchor='nw', expand=True)

    route = tk.Label(root, text=f'Current route: {start} to {end}', font=('Helvetica', '15', 'bold'),fg = 'black')
    route.place(x=10, y=655)

    depthLabel = tk.Label(root, text= '', font = ('Helvetica', '15'), fg = 'red')
    depthLabel.place(x = len(f'Current route: {start} to {end}') * 10 + 50, y = 655)

    default = tk.StringVar()
    default.set("Choose an \nalgorithm")
    drop = tk.OptionMenu(root, default, 'Dijkstra', 'A*', 'DFS', 'ID-DFS', 'Bidirectional Dijkstra', 'Reverse A*',
                         'temp')
    drop.config(width=15, height=3, bg='white', font=('Helvetica', '10', 'bold'))
    drop.place(x=1202, y=0)

    run = tk.Button(root, text='RUN ALGORITHM')
    run.config(width=17, height=2, bg='green', font=('Helvetica', '10', 'bold'), fg='white', command=display_selected)
    run.place(x=1203, y=500)

    stop = tk.Button(root, text = 'PAUSE ALGORITHM')
    stop.config(width = 18, height = 2, bg = 'red', font = ('Helvetica', '10', 'bold'), fg = 'white', command = stoprun)
    stop.place(x = 1200, y = 550)

    routes = draw_map()
    x1, y1 = get_coords(nodes[startID])
    x2, y2 = get_coords(nodes[endID])
    canvas.create_oval(x1-5, y1-5, x1+5, y1+5, fill = 'green')
    canvas.create_oval(x2-5, y2-5, x2+5, y2+5, fill = 'red')
    canvas.update()
    root.mainloop()


if __name__ == "__main__":
    main()
