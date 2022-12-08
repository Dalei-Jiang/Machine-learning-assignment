# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021
# import sys
"""
This is the main entry point for MP3. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)
# import heapq 
# from collections import deque
# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
# Note that if you want to test one of your search methods, please make sure to return a blank list
#  for the other search methods otherwise the grader will not crash.
class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): DISTANCE(i, j)
                for i, j in self.cross(objectives)
            }
    # Calculate the distance between to points
    
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root

    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a)
        rb = self.resolve(b)
        if ra == rb:
            return False
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

def DISTANCE(i, j):
    return abs(i[0]-j[0])+abs(i[1]-j[1])

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    final_path = []
    start_point = maze.start # pick the starting point
    path_list = [start_point]
    destination = maze.waypoints[0] # In this part, we can assume that there is only one destinaton
    
    passed_points = {start_point:path_list}  # Used to store all the points we have passed, and their distance cost
    current_group = [start_point]
    successor_group = [] # Store the next level's points
    reach = False
    while (not reach):
        for p in current_group:
            neighbor = maze.neighbors(p[0],p[1])
            for n in neighbor:
                if n == destination:
                    final_path = passed_points[p]
                    final_path.append(n)
                    reach = True
                    break
                if n not in passed_points:
                    parent_path = passed_points[p].copy()
                    passed_points[n] = parent_path
                    passed_points[n].append(n)
                    successor_group.append(n)
            if (reach):
                break
        current_group = successor_group
        successor_group = []
    return final_path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    final_path = []
    start_point = maze.start
    path_list = [start_point]
    destination = maze.waypoints[0]
    passed_points = {start_point:path_list}
    current_group = [(start_point,0)]
    reach = False
    while (not reach):
        ex_point = (-1,-1)
        min_dist = -1
        min_way = 0
        for p in current_group:
            H = p[1] + abs(p[0][0]-destination[0]) + abs(p[0][1]-destination[1])
            if (H < min_dist or min_dist==-1):
                min_dist =  H
                min_way = p[1]
                ex_point = p[0]
                remove_item = p
        neighbor = maze.neighbors(ex_point[0],ex_point[1])
        current_group.remove(remove_item)
        for n in neighbor:
            if n == destination:
                final_path = passed_points[remove_item[0]]
                final_path.append(n)
                reach = True
                break
            if (n not in passed_points):
                parent_path = passed_points[remove_item[0]].copy()
                passed_points[n] = parent_path
                passed_points[n].append(n)    
                current_group.append((n, min_way + 1))               
        if (reach):
            break
    return final_path

def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
       
    final_path = []
    start_point = maze.start
    destinations = maze.waypoints
    state = (start_point, destinations)
    passed_points = [(state, 0)]
    state_gn = {state:0}
    state_fn = {state:0}
    paths = {state:[start_point]}
    mst_dic = {}
    processing_queue = [state]
    reach = False
    counter = 0
    while (not reach and counter < 10000):
        counter += 1
        min_f = -1
        for s in processing_queue:
            f = state_fn[s]
            if (f < min_f or min_f == -1):
                expanding_state = s
                min_f = f
        neignborhoods = maze.neighbors(expanding_state[0][0], expanding_state[0][1])
        passed_points.append((expanding_state, min_f))
        processing_queue.remove(expanding_state)
        for n in neignborhoods:
            new_des = expanding_state[1]
            if n in expanding_state[1]:
                new_des = tuple(set(new_des) - {n})
            new_state = (n, new_des)
            if len(new_des) == 0:
                final_path = paths[expanding_state]
                final_path.append(n)
                reach = True
                break
            f1 = state_gn[expanding_state] + 1
            min_dis = -1
            for i in new_state[1]:
                dist = DISTANCE(i, new_state[0])
                if (dist < min_dis or min_dis == -1):
                    min_dis = dist
            f2 = min_dis
            if (new_state[1] in mst_dic.keys()):
                f3 = mst_dic[new_state[1]]
            else:
                mst = MST(new_state[1])
                f3 = mst.compute_mst_weight()
                mst_dic[new_state[1]] = f3
            fnew =  f1 + f2 + f3
            
            if new_state not in passed_points:
                if (new_state not in state_fn.keys() or (new_state in state_fn.keys() and state_fn[new_state] >= fnew)):
                    state_gn[new_state] = state_gn[expanding_state]+1
                    state_fn[new_state] = fnew
                    passed_points.append(new_state)
                    new_path = paths[expanding_state].copy()
                    new_path.append(new_state[0])
                    paths[new_state] = new_path
                    processing_queue.append(new_state)
        if (reach):
            break
    return final_path


def fast(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    final_path = []
    start_point = maze.start
    destinations = maze.waypoints
    state = (start_point, destinations)
    passed_points = [(state, 0)]
    state_gn = {state:0}
    state_fn = {state:0}
    paths = {state:[start_point]}
    mst_dic = {}
    processing_queue = [state]
    reach = False
    counter = 0
    while (not reach and counter < 10000000):
        counter+=1
        min_f = -1
        for s in processing_queue:
            f = state_fn[s]
            if (f < min_f or min_f == -1):
                expanding_state = s
                min_f = f
        neignborhoods = maze.neighbors(expanding_state[0][0], expanding_state[0][1])
        passed_points.append((expanding_state, min_f))
        processing_queue.remove(expanding_state)
        for n in neignborhoods:
            new_des = expanding_state[1]
            if n in expanding_state[1]:
                new_des = tuple(set(new_des) - {n})
            new_state = (n, new_des)
            if len(new_des) == 0:
                final_path = paths[expanding_state]
                final_path.append(n)
                reach = True
                break
            # f1 = state_gn[expanding_state] + 1
            min_dis = -1
            for i in new_state[1]:
                dist = DISTANCE(i, new_state[0])
                if (dist < min_dis or min_dis == -1):
                    min_dis = dist
            f2 = min_dis
            if (new_state[1] in mst_dic.keys()):
                f3 = mst_dic[new_state[1]]
            else:
                mst = MST(new_state[1])
                f3 = mst.compute_mst_weight()
                mst_dic[new_state[1]] = f3
            fnew =  f2 + f3
            
            if new_state not in passed_points:
                if (new_state not in state_fn.keys() or (new_state in state_fn.keys() and state_fn[new_state] > fnew)):
                    state_gn[new_state] = state_gn[expanding_state]+1
                    state_fn[new_state] = fnew
                    passed_points.append(new_state)
                    new_path = paths[expanding_state].copy()
                    new_path.append(new_state[0])
                    paths[new_state] = new_path
                    processing_queue.append(new_state)
        if (reach):
            break
    return final_path

