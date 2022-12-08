def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    unreached_dic = {}
    processing = []
    start = maze.start
    destinations = maze.waypoints
    G = 0
    F1 = F1(start, destination)
    
    info_list = []
    
    
def f1(n, unreached):
    minh = -1
    for u in unreached:
        h = DISTANCE(n, u)
        if (h < minh or minh == -1):
            minh = h
    return minh