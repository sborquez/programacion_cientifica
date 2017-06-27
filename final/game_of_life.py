import numba
import numpy as np
import matplotlib.pyplot as plt

def init_universe(rows, cols, cells):
    universe = np.zeros((rows+2, cols+2), bool)
    universe[cells[0],cells[1]] = True
    return universe

def init_universe_random(rows, cols, cells):
    universe = np.zeros((rows+2, cols+2), bool)
    r = np.random.randint(1,rows+1,cells)
    c = np.random.randint(1,cols+1,cells)
    universe[r,c] = True
    return universe

def show(universe):
    rows,cols = universe.shape
    plt.figure(figsize=(10,10))
    plt.imshow(universe[1:rows-1,1:cols-1], cmap='bwr')
    plt.axis('off')
    plt.grid()
    plt.show()

def nn_step(universe, new_universe, rule):
    rows, cols = universe.shape
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            new_universe[i,j] = np.sum(universe[i-1:i+2,j-1:j+2] * rule)
    
    return new_universe


def nn_evolve(universe, rule, num, t):
    t0 = 0
    rows, cols = universe.shape
    new_universe = np.zeros((rows,cols), int)
    while t0 < t:
        new_universe = step(universe, new_universe, rule)
        universe = np.zeros((rows,cols), bool)
        for value in num:
            universe = np.logical_or(new_universe == value, universe)
        t0 += 1
    return universe 

@numba.jit('int64[:,:] (boolean[:,:], int64[:,:], int64[:,:])', nopython=True)
def step(universe, new_universe, rule):
    rows, cols = universe.shape
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            new_universe[i,j] = np.sum(universe[i-1:i+2,j-1:j+2] * rule)
    
    return new_universe

@numba.jit('boolean[:,:] (boolean[:,:], int64[:,:], int64[:,:], int64)')
def evolve(universe, rule, num, t):
    t0 = 0
    rows, cols = universe.shape
    new_universe = np.zeros((rows,cols), int)
    while t0 < t:
        new_universe = step(universe, new_universe, rule)
        universe = np.zeros((rows,cols), bool)
        for value in num:
            universe = np.logical_or(new_universe == value, universe)
        t0 += 1
    return universe 

rules = {
    "Standard" : (np.array([[1,1,1],[1,-9,1],[1,1,1]], int), np.array([3,-6,-7], int)),
    "Diagonales" : (np.array([[1,0,1],[0,-9,0],[1,0,1]], int), np.array([3,-6,-7], int)),
    "Cruz" : (np.array([[0,1,0],[1,-9,1],[0,1,0]], int), np.array([3,-6,-7], int)),
    "Fast Grow": (np.array([[1,1,1],[1,-9,1],[1,1,1]], int), np.array([3,4,5,6,7,-5,-6,-7,-8], int)),
    "Strong": (np.array([[1,1,1],[1,-9,1],[1,1,1]], int), np.array([3,4,-5,-6,-7,-8], int)),
    "xD": (np.array([[1,1,1],[1,-9,1],[1,1,1]], int), np.array([1,8, -1], int))
}