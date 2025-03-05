from GeneticSolver import GeneticSolver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def branin(pos, a = 1, b = 5.1/(4*np.pi**2), c = 5/np.pi, r = 6, s = 10, t = 1/(8*np.pi)):
    return a * (pos[1] - b * pos[0]**2 + c * pos[0] - r)**2 + s*(1-t)*np.cos(pos[0]) + s

def easom(pos):
    return -np.cos(pos[0])*np.cos(pos[1]) * np.e**(-(pos[0]-np.pi)**2-(pos[1]-np.pi)**2)

def goldstein_price(pos):
    return (
        1 + (pos[0]+pos[1]+1)**2 * (19 - 14*pos[0] + 3*pos[0]**2 - 14*pos[1] + 6*pos[0]*pos[1] + 3*pos[1]**2)
    )*(
        30 + (2*pos[0] - 3*pos[1])**2 * (18-32*pos[0] + 12*pos[0]**2 + 48*pos[1] - 36*pos[0]*pos[1] + 27*pos[1]**2)
    )

def camel(pos):
    return (4 - 2.1*pos[0]**2 + pos[0]**4/3)*pos[1]**2 + pos[0]*pos[1] + (-4+4*pos[1]**2)*pos[1]**2

"""
gs_branin_1 = GeneticSolver(branin, 10, 20, 2, [[-5, 15],[0, 15]], 0.1, 0.1, True)

gs_branin_1.anisolve(100)
gs_branin_1.reset()
gs_branin_1.solve_stats(100, True, 5, 0.001, True)

gs_branin_2 = GeneticSolver(branin, 5, 5, 2, [[-5, 15],[0, 15]], 0.1, 0.1, True)

gs_branin_2.anisolve(50)
gs_branin_2.reset()
gs_branin_2.solve_stats(50, True, 5, 0.001, True)

stat_1 = []
stat_2 = []
for _ in tqdm(range(100), desc = "branin compare"):
    gs_branin_1.reset()
    stat_1.append([gs_branin_1.solve_stats(100, False, 5, 0, False)])
    gs_branin_2.reset()
    stat_2.append([gs_branin_2.solve_stats(50, False, 5, 0)])
stat_1 = np.mean(stat_1, axis = 0)[0]
stat_2 = np.mean(stat_2, axis = 0)[0]

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(range(len(stat_1)), stat_1, "b", label = "перший набір")
ax.plot(range(len(stat_2)), stat_2, "r", label = "другий набір")
ax.set_xlabel("iteration")
ax.set_ylabel("best value")
ax.legend()
plt.show()
"""

gs_easom_1 = GeneticSolver(easom, 100, 200, 2, [[-100, 100],[-100, 100]], 0.25, 0.25, True)
gs_easom_1.anisolve(100)
gs_easom_1.reset()
gs_easom_1.solve_stats(100,True,5,0.001, True)

gs_easom_2 = GeneticSolver(easom, 25, 25, 2, [[-100, 100],[-100, 100]], 0.25, 0.25, True)
gs_easom_2.anisolve(50)
gs_easom_2.reset()
gs_easom_2.solve_stats(50,True,5,0.001, True)

gs_easom_3 = GeneticSolver(easom, 25, 25, 2, [[-100, 100],[-100, 100]], 0.05, 0.05, True)
gs_easom_3.anisolve(50)
gs_easom_3.reset()
gs_easom_3.solve_stats(50,True,5,0.001, True)

stat_1 = []
stat_2 = []
stat_3 = []
for _ in tqdm(range(100), desc = "easom compare"):
    gs_easom_1.reset()
    stat_1.append([gs_easom_1.solve_stats(100, False, 5, 0)])
    gs_easom_2.reset()
    stat_2.append([gs_easom_2.solve_stats(50, False, 5, 0)])
    gs_easom_3.reset()
    stat_3.append([gs_easom_3.solve_stats(50, False, 5, 0)])
stat_1 = np.mean(stat_1, axis = 0)[0]
stat_2 = np.mean(stat_2, axis = 0)[0]
stat_3 = np.mean(stat_3, axis = 0)[0]

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(range(len(stat_1)), stat_1, "b", label = "перший набір")
ax.plot(range(len(stat_2)), stat_2, "r", label = "другий набір")
ax.plot(range(len(stat_3)), stat_3, "g", label = "третій набір")
ax.set_xlabel("iteration")
ax.set_ylabel("best value")
ax.legend()
plt.show()