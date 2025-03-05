from GeneticSolver import GeneticSolver
import numpy as np


def branin(pos, a = 1, b = 5.1/(4*np.pi**2), c = 5/np.pi, r = 6, s = 10, t = 1/(8*np.pi)):
    return a * (pos[1] - b * pos[0]**2 + c * pos[0] - r)**2 + s*(1-t)*np.cos(pos[0]) + s

gs = GeneticSolver(branin, 100, 200, 2, [[-5, 10],[0, 15]], 0.1, 0.05, True)

print(gs.solve(1000, epsilon_timeout=5, epsilon=0.1))