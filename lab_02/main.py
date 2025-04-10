from PSO import PSOSolver
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def test_mean(object, iterations, tests, desc = "test_mean"):
    output = []
    for _ in tqdm(range(tests), desc = desc):
        object.reset()
        output.append([
            object.solve_stats(iterations)[2]
        ])
    return np.mean(output, axis=0)[0]

def test_time(object, iterations, tests, desc = "test_time"):
    output = []
    for _ in tqdm(range(tests), desc = desc):
        object.reset()
        output.append([
            object.solve_time(iterations)[2]
        ])
    return np.mean(output, axis=0)[0]

def ackley(pos):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (pos[0] ** 2 + pos[1] ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * pos[0]) + np.cos(2 * np.pi * pos[1]))) + np.e + 20


def rosenbroke(pos):
    output = 0
    for i in range(len(pos) - 1):
        output += 100 * (pos[i + 1] - pos[i] ** 2) ** 2 + (pos[i] - 1) ** 2
    return output


def cross_in_tray(pos):
    return -0.0001 * (abs(np.sin(pos[0]) * np.sin(pos[1]) * np.exp(
        abs(100 - np.sqrt(pos[0] ** 2 + pos[1] ** 2) / np.pi))) + 1) ** 0.1


def holder_table(pos):
    return -abs(np.sin(pos[0]) * np.cos(pos[1]) * np.exp(abs(1 - np.sqrt(pos[0] ** 2 + pos[1] ** 2) / np.pi)))


def mccormick(pos):
    return np.sin(pos[0] + pos[1]) + (pos[0] - pos[1]) ** 2 - 1.5 * pos[0] + 2.5 * pos[1] + 1


def styrblinski_tang(pos):
    output = 0
    for i in range(len(pos)):
        output += pos[i] ** 4 - 16 * pos[i] ** 2 + 5 * pos[i]
    return output/2


ackley_pso_1 = PSOSolver({
    "a1": 0.1,#acceleration number
    "a2": 0.2,#acceleration number
    "pop_size": 50,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-5,-5]),#vector of minimum positions
    "pos_max": np.array([5,5]),#vector of maximum positions
    "speed_min": np.array([-1,-1]),#vector of min speed
    "speed_max": np.array([1,1]),#vector of max speed
}, ackley, True)

ackley_pso_1.anisolve()

ackley_pso_2 = PSOSolver({
    "a1": 0.01,#acceleration number
    "a2": 0.02,#acceleration number
    "pop_size": 50,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-5,-5]),#vector of minimum positions
    "pos_max": np.array([5,5]),#vector of maximum positions
    "speed_min": np.array([-0.1,-0.1]),#vector of min speed
    "speed_max": np.array([0.1,0.1]),#vector of max speed
}, ackley, True)

ackley_pso_2.anisolve()

a_1 = test_mean(ackley_pso_1, 1000, 100, "ackley_1")
a_2 = test_mean(ackley_pso_2, 1000, 100, "ackley_2")
plt.plot(range(len(a_1)), a_1, label = "ackley 1")
plt.plot(range(len(a_2)), a_2, label = "ackley 2")
plt.legend()
plt.yscale("log")
plt.show()