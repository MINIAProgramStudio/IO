from PSO import PSOSolver
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

TESTS = 100
ITERATIONS = 250

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

""""""
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

ackley_pso_1.anisolve(save=True)

ackley_pso_2 = PSOSolver({
    "a1": 0.01,#acceleration number
    "a2": 0.02,#acceleration number
    "pop_size": 50,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-5,-5]),#vector of minimum positions
    "pos_max": np.array([5,5]),#vector of maximum positions
    "speed_min": np.array([-1,-1]),#vector of min speed
    "speed_max": np.array([1,1]),#vector of max speed
}, ackley, True)

ackley_pso_2.anisolve(save=True)

a_1 = test_mean(ackley_pso_1, ITERATIONS, TESTS, "ackley_1")
a_2 = test_mean(ackley_pso_2, ITERATIONS, TESTS, "ackley_2")
plt.plot(range(len(a_1)), a_1, label = "ackley 1")
plt.plot(range(len(a_2)), a_2, label = "ackley 2")
plt.legend()
plt.yscale("log")
plt.show()

rosenbroke_pso_1 = PSOSolver({
    "a1": 10**7,#acceleration number
    "a2": 2*10**7,#acceleration number
    "pop_size": 50,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-10**9,-10**9]),#vector of minimum positions
    "pos_max": np.array([10**9,10**9]),#vector of maximum positions
    "speed_min": np.array([-10**8,-10**8]),#vector of min speed
    "speed_max": np.array([10**8,10**8]),#vector of max speed
}, rosenbroke, True)

rosenbroke_pso_1.anisolve(save=True)

rosenbroke_pso_2 = PSOSolver({
    "a1": 10**6,#acceleration number
    "a2": 2*10**6,#acceleration number
    "pop_size": 50,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-10**9,-10**9]),#vector of minimum positions
    "pos_max": np.array([10**9,10**9]),#vector of maximum positions
    "speed_min": np.array([-10**7,-10**7]),#vector of min speed
    "speed_max": np.array([10**7,10**7]),#vector of max speed
}, rosenbroke, True)

rosenbroke_pso_2.anisolve(save=True)

rosenbroke_pso_3 = PSOSolver({
    "a1": 10**6,#acceleration number
    "a2": 2*10**6,#acceleration number
    "pop_size": 200,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-10**9,-10**9]),#vector of minimum positions
    "pos_max": np.array([10**9,10**9]),#vector of maximum positions
    "speed_min": np.array([-10**7,-10**7]),#vector of min speed
    "speed_max": np.array([10**7,10**7]),#vector of max speed
}, rosenbroke, True)

rosenbroke_pso_3.anisolve(save=True)


r_1 = test_mean(rosenbroke_pso_1, ITERATIONS, TESTS, "rosenbroke_1")
r_2 = test_mean(rosenbroke_pso_2, ITERATIONS, TESTS, "rosenbroke_2")
r_3 = test_mean(rosenbroke_pso_3, ITERATIONS, TESTS, "rosenbroke_3")
plt.plot(range(len(r_1)), r_1, label = "rosenbroke 1")
plt.plot(range(len(r_2)), r_2, label = "rosenbroke 2")
plt.plot(range(len(r_3)), r_3, label = "rosenbroke 3")
plt.legend()
plt.yscale("log")
plt.show()


cross_in_tray_pso_1 = PSOSolver({
    "a1": 0.1,#acceleration number
    "a2": 0.2,#acceleration number
    "pop_size": 50,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-10,-10]),#vector of minimum positions
    "pos_max": np.array([10,10]),#vector of maximum positions
    "speed_min": np.array([-1,-1]),#vector of min speed
    "speed_max": np.array([1,1]),#vector of max speed
}, cross_in_tray, True)

cross_in_tray_pso_1.anisolve(save=True)

cross_in_tray_pso_2 = PSOSolver({
    "a1": 0.1,#acceleration number
    "a2": 0.2,#acceleration number
    "pop_size": 100,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-10,-10]),#vector of minimum positions
    "pos_max": np.array([10,10]),#vector of maximum positions
    "speed_min": np.array([-1,-1]),#vector of min speed
    "speed_max": np.array([1,1]),#vector of max speed
}, cross_in_tray, True)

cross_in_tray_pso_2.anisolve(save=True)

c_1 = test_mean(cross_in_tray_pso_1, int(ITERATIONS/10), TESTS, "cross_in_tray_1")
c_2 = test_mean(cross_in_tray_pso_2, int(ITERATIONS/10), TESTS, "cross_in_tray_2")
plt.plot(range(len(c_1)), c_1, label = "cross_in_tray 1")
plt.plot(range(len(c_2)), c_2, label = "cross_in_tray 2")
plt.legend()
plt.show()


holder_table_pso_1 = PSOSolver({
    "a1": 0.1,#acceleration number
    "a2": 0.2,#acceleration number
    "pop_size": 50,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-10,-10]),#vector of minimum positions
    "pos_max": np.array([10,10]),#vector of maximum positions
    "speed_min": np.array([-1,-1]),#vector of min speed
    "speed_max": np.array([1,1]),#vector of max speed
}, holder_table, True)

holder_table_pso_1.anisolve(save=True)

holder_table_pso_2 = PSOSolver({
    "a1": 0.1,#acceleration number
    "a2": 0.2,#acceleration number
    "pop_size": 100,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-10,-10]),#vector of minimum positions
    "pos_max": np.array([10,10]),#vector of maximum positions
    "speed_min": np.array([-1,-1]),#vector of min speed
    "speed_max": np.array([1,1]),#vector of max speed
}, holder_table, True)

holder_table_pso_2.anisolve(save=True)

h_1 = test_mean(holder_table_pso_1, ITERATIONS, TESTS, "holder_table_1")
h_2 = test_mean(holder_table_pso_2, ITERATIONS, TESTS, "holder_table_2")
plt.plot(range(len(h_1)), h_1, label = "holder_table 1")
plt.plot(range(len(h_2)), h_2, label = "holder_table 2")
plt.legend()
plt.show()


mccormick_pso_1 = PSOSolver({
    "a1": 0.1,#acceleration number
    "a2": 0.2,#acceleration number
    "pop_size": 50,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-1.5,-3]),#vector of minimum positions
    "pos_max": np.array([4,4]),#vector of maximum positions
    "speed_min": np.array([-1,-1]),#vector of min speed
    "speed_max": np.array([1,1]),#vector of max speed
}, mccormick, True)

mccormick_pso_1.anisolve(save=True)

mccormick_pso_2 = PSOSolver({
    "a1": 1,#acceleration number
    "a2": 2,#acceleration number
    "pop_size": 50,#population size
    "dim": 2,#dimensions
    "pos_min": np.array([-1.5,-3]),#vector of minimum positions
    "pos_max": np.array([4,4]),#vector of maximum positions
    "speed_min": np.array([-1,-1]),#vector of min speed
    "speed_max": np.array([1,1]),#vector of max speed
}, mccormick, True)

mccormick_pso_2.anisolve(save=True)

m_1 = test_mean(mccormick_pso_1, ITERATIONS, TESTS, "mccormick_1")
m_2 = test_mean(mccormick_pso_2, ITERATIONS, TESTS, "mccormick_2")
plt.plot(range(len(m_1)), m_1, label = "mccormick 1")
plt.plot(range(len(m_2)), m_2, label = "mccormick 2")
plt.legend()
plt.show()