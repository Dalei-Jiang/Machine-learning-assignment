import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})
    
###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
# Part 1 
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if side:    # remember, side -> MIN, !side -> MAX
        value = math.inf
    else:
        value = -math.inf
    moveList = []
    moveTree = {}
    if depth > 1:
        moves = generateMoves(side, board, flags)
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            newdepth = depth-1
            subvalue, submoveList, submoveTree = minimax(newside, newboard, newflags, newdepth)
            movecode = encode(move[0], move[1], move[2])
            moveTree[movecode] = submoveTree # Connect the subtree to the current node
            if (side and subvalue < value) or (not side and subvalue > value):
                value = subvalue
                moveList = submoveList
                moveList.insert(0,move)
    # When depth = 1, we reach the last layer
    elif depth == 1:
        moves = generateMoves(side, board, flags)
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            subvalue = evaluate(newboard)
            movecode = encode(move[0], move[1], move[2])
            moveTree[movecode] = {}
            if (side and subvalue < value) or (not side and subvalue > value):
                value = subvalue
                greatmove = move
        moveList = [greatmove]
    # else:
    #     value = evaluate(board)
    #     moveList = []
    #     moveTree = {}
    return value, moveList, moveTree

###########################################################################################
def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if side:    # remember, side -> MIN, !side -> MAX
        value = math.inf
    else:
        value = -math.inf
    moveList = []
    moveTree = {}
    if depth > 1:
        moves = generateMoves(side, board, flags)
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            newdepth = depth-1
            subvalue, submoveList, submoveTree = alphabeta(newside, newboard, newflags, newdepth, alpha, beta)
            movecode = encode(move[0], move[1], move[2])
            moveTree[movecode] = submoveTree # Connect the subtree to the current node
            if (side and subvalue < value) or (not side and subvalue > value): 
                value = subvalue
                moveList = submoveList
                moveList.insert(0,move)
            if side and value < beta:
                beta = value
            elif not side and value > alpha:
                alpha = value
            if alpha >= beta:
                return value, moveList, moveTree # End the function when a >= b
    elif depth == 1:
        moves = generateMoves(side, board, flags)
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            newdepth = depth-1
            subvalue = evaluate(newboard)
            movecode = encode(move[0], move[1], move[2])
            moveTree[movecode] = {} # Connect the subtree to the current node
            if (side and subvalue < value) or (not side and subvalue > value): 
                value = subvalue
                greatmove = move
            moveList = [greatmove]
            if side and value < beta:
                beta = value
            elif not side and value > alpha:
                alpha = value
            if alpha >= beta:
                return value, moveList, moveTree
    # else:
    #     value = evaluate(board)
    #     moveList = []
    #     moveTree = {}
    return value, moveList, moveTree
        
###########################################################################################
def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    # if side:    # remember, side -> MIN, !side -> MAX
    #     value = math.inf
    # else:
    #     value = -math.inf
    # moveList = []
    # moveTree = {}
    # Firstmoves = [ move for move in generateMoves(side, board, flags) ]
    # compdict = {}
    # Listdict = {}
    # for move in Firstmoves:
    #     newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2]) # The board after first layer
    #     valueaccount = 0
    #     for i in range(breadth):
    #         rand_value, rand_moveList, rand_moveTree = recuriser_stochastic(1, breadth, depth, newside, newboard, newflags, chooser)
    #         rand_moveList.insert(0, move)
    #         movecode = encode(move[0], move[1], move[2])
    #         valueaccount += rand_value
    #         moveTree[movecode] = rand_moveTree
    #         if (side and rand_value < value) or (not side and rand_value > value): 
    #             great_moveList = rand_moveList
    #     average_value = valueaccount/breadth
    #     compdict[movecode] = average_value
    #     Listdict[movecode] = great_moveList
    # if side:
    #     for key in compdict.keys():
    #         if compdict[key] < value:
    #             value = compdict[key]
    #             moveList = Listdict[key]
    # else:
    #     for key in compdict.keys():
    #         if compdict[key] > value:
    #             value = compdict[key]
    #             moveList = Listdict[key]
    # return value, moveList, moveTree


    if side:    # remember, side -> MIN, !side -> MAX
        value = math.inf
    else:
        value = -math.inf
    moveList = []
    moveTree = {}
    Firstmoves = [ move for move in generateMoves(side, board, flags) ]
    compdict = {}
    Listdict = {}
    
    for move in Firstmoves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2]) # The board after first layer
        rand_value, rand_moveList, rand_moveTree = recuriser_stochastic(1, breadth, depth, newside, newboard, newflags, chooser)
        rand_moveList.insert(0, move)
        movecode = encode(move[0], move[1], move[2])
        moveTree[movecode] = rand_moveTree
        compdict[movecode] = rand_value
        Listdict[movecode] = rand_moveList
    if side:
        for key in compdict.keys():
            if compdict[key] < value:
                value = compdict[key]
                moveList = Listdict[key]
    else:
        for key in compdict.keys():
            if compdict[key] > value:
                value = compdict[key]
                moveList = Listdict[key]
    return value, moveList, moveTree

###########################################################################################
def recuriser_stochastic(process, breadth, depth, side, board, flags, chooser):
    '''
    Randomly choose paths and reach the leaf to get the leaf node value, and return the average value.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      process (int >=0): the layer we current locates at
      breadth: number of different paths 
      depth (int >=0): depth of the search (number of moves)      
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    if side:    # remember, side -> MIN, !side -> MAX
        value = math.inf
    else:
        value = -math.inf
    moves = [ move for move in generateMoves(side, board, flags) ]
    moveList = []
    moveTree = {}
    # print(process)
    if process == depth:
        value = evaluate(board)
        moveList = []
        moveTree = {}
    elif process == 1: # the recursion is just at the beginning
        value_counter = 0
        for i in range(breadth):
            move = chooser(moves)  
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            newprocess = process+1
            randvalue, randmoveList, randmoveTree = recuriser_stochastic(newprocess, breadth, depth, newside, newboard, newflags, chooser)
            movecode = encode(move[0], move[1], move[2])
            moveTree[movecode] = randmoveTree # Connect the subtree to the current node
            value_counter += randvalue
            if (side and randvalue < value) or (not side and randvalue > value): 
                greatList = randmoveList
        value = value_counter/breadth
        moveList = greatList
    elif process > 1:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        newprocess = process+1
        randvalue, randmoveList, randmoveTree = recuriser_stochastic(newprocess, breadth, depth, newside, newboard, newflags, chooser)
        movecode = encode(move[0], move[1], move[2])
        moveTree[movecode] = randmoveTree
        value = randvalue
        randmoveList.insert(0,move)
        moveList = randmoveList
    return value, moveList, moveTree


    # if side:    # remember, side -> MIN, !side -> MAX
    #     value = math.inf
    # else:
    #     value = -math.inf
    # moves = [ move for move in generateMoves(side, board, flags) ]
    # moveList = []
    # moveTree = {}
    # # print(process)
    # if process == depth:
    #     value = evaluate(board)
    #     moveList = []
    #     moveTree = {}
    # elif process >= 1:
    #     move = chooser(moves)
    #     newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
    #     newprocess = process+1
    #     randvalue, randmoveList, randmoveTree = recuriser_stochastic(newprocess, breadth, depth, newside, newboard, newflags, chooser)
    #     movecode = encode(move[0], move[1], move[2])
    #     moveTree[movecode] = randmoveTree
    #     value = randvalue
    #     randmoveList.insert(0,move)
    #     moveList = randmoveList
    # return value, moveList, moveTree
        
    # if side:    # remember, side -> MIN, !side -> MAX
    #     value = math.inf
    # else:
    #     value = -math.inf
    # moves = [ move for move in generateMoves(side, board, flags) ]
    # moveList = []
    # moveTree = {}
    # # print(process)
    # if process == depth:
    #     return (evaluate(board), [], {})
    # elif process >= 1:
    #     move = chooser(moves)
    #     newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
    #     newprocess = process+1
    #     randvalue, randmoveList, randmoveTree = recuriser_stochastic(newprocess, breadth, depth, newside, newboard, newflags, chooser)
    #     movecode = encode(move[0], move[1], move[2])
    #     moveTree[movecode] = randmoveTree
    #     value = randvalue
    #     randmoveList.insert(0,move)
    #     moveList = randmoveList
    # elif process == 0:
    #     value = evaluate(board)
    # return value, moveList, moveTree