from random import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class GeneticSolver:
    def __init__(self, func, pop_size, children, dimensions, minmax, mutation_prob, mutation_pow, seeking_min = False):
        # запам'ятовуємо гіперпараметри
        self.func = func
        self.pop_size = pop_size
        self.children = children
        self.dimensions = dimensions
        self.minmax = minmax
        self.mutation_prob = mutation_prob
        self.mutation_pow = mutation_pow
        self.seeking_min = seeking_min

        #створюємо хромосоми
        self.pop = [[random()*(minmax[d][1] - minmax[d][0]) + minmax[d][0] for d in range(dimensions)] for _ in range(pop_size)]

    def reset(self):
        self.pop = [[random() * (self.minmax[d][1] - self.minmax[d][0]) + self.minmax[d][0] for d in range(self.dimensions)] for _ in range(self.pop_size)]

    def select(self):
        self.pop = sorted(self.pop, key = self.func, reverse = not self.seeking_min) # відсортувати хромосоми за фітнес-функцією
        self.pop = self.pop[:self.pop_size] # залишити лише найкращі хромосоми у кількості pop_size штук
        return self.pop[0] # повернути найкращу хромосому

    def select_mp(self):
        with ProcessPoolExecutor(max_workers=2) as executor:
            fitness_values = list(executor.map(self.func, self.pop))

        # Поєднати хромосоми з їхніми значеннями фітнес-функції
        combined = list(zip(self.pop, fitness_values))

        # Відсортувати за значенням фітнес-функції
        combined.sort(key=lambda x: x[1], reverse=not self.seeking_min)

        # Оновити популяцію, залишивши лише найкращі хромосоми
        self.pop = [chrom for chrom, _ in combined[:self.pop_size]]

        return self.pop[0]  # Повернути найкращу хромосому

    def crossover(self):
        for _ in range(self.children): # зробити children нових хромосом
            # псевдовипадкова генерація індексів батьків
            a = int(random()*self.pop_size)
            b = int(random()*(self.pop_size-1))
            if b >= a: b += 1
            # створити хромосому та додати її до популяції
            self.pop.append([
                random()*abs(self.pop[a][d] - self.pop[b][d]) + min(self.pop[a][d], self.pop[b][d]) for d in range(self.dimensions)
            ])

    def mutate(self):
        for i in range(len(self.pop)): # для кожної хромосоми
            for d in range(self.dimensions): # для кожної координати (для кожного гена)
                if random() < self.mutation_prob: # з імовірністю mutation_prob
                    self.pop[i][d] = self.pop[i][d] + ((-1)**(int(random()*2))) * self.mutation_pow * random() * (self.minmax[d][1] - self.minmax[d][0]) #
                    self.pop[i][d] = min(self.minmax[d][1], max(self.minmax[d][0], self.pop[i][d]))

    def iter(self):
        self.crossover()
        self.mutate()
        self.select()

    def solve(self, iterations, progressbar = True, epsilon_timeout = float("inf"), epsilon = 0):
        if progressbar:
            iterator = tqdm(range(iterations))
        else:
            iterator = range(iterations)
        epsilon_timeout_counter = 0

        for _ in iterator:
            old_best = self.pop[0]
            self.iter()

            if self.func(self.pop[0]) - self.func(old_best) >= epsilon and not self.seeking_min:
                epsilon_timeout_counter = 0
            elif self.func(old_best) - self.func(self.pop[0]) >= epsilon and self.seeking_min:
                epsilon_timeout_counter = 0
            else:
                epsilon_timeout_counter += 1
                if epsilon_timeout_counter > epsilon_timeout:
                    break
        return (self.func(self.pop[0]), self.pop[0])

    def solve_stats(self, iterations, progressbar = True, epsilon_timeout = float("inf"), epsilon = 0, show = False):
        y = []
        if progressbar:
            iterator = tqdm(range(iterations))
        else:
            iterator = range(iterations)
        epsilon_timeout_counter = 0
        self.select()
        y.append(self.func(self.pop[0]))
        for _ in iterator:
            old_best = self.pop[0]


            self.crossover()
            self.mutate()
            self.select()
            y.append(self.func(self.pop[0]))

            if self.func(self.pop[0]) - self.func(old_best) >= epsilon and not self.seeking_min:
                epsilon_timeout_counter = 0
            elif self.func(old_best) - self.func(self.pop[0]) >= epsilon and self.seeking_min:
                epsilon_timeout_counter = 0
            else:
                epsilon_timeout_counter += 1
                if epsilon_timeout_counter > epsilon_timeout:
                    break
        self.select()
        y.append(self.func(self.pop[0]))
        if show:
            x = range(len(y))
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(x[:],y[:])
            # plt.yscale("log")
            ax.set_xlabel("iteration")
            ax.set_ylabel("best value")
            plt.show()
        return y

    def anisolve(self, iterations, save = False):
        if self.dimensions != 2:
            raise Exception("GeneticSolver.anisolve can visualise only 2d functions")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d",computed_zorder=False)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")

        func_x = np.linspace(self.minmax[0][0], self.minmax[0][1], 106)
        func_y = np.linspace(self.minmax[1][0], self.minmax[1][1], 106)
        FUNC_X, FUNC_Y = np.meshgrid(func_x, func_y)
        FUNC_Z = self.func([FUNC_X, FUNC_Y])

        ax.plot_surface(FUNC_X, FUNC_Y, FUNC_Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, zorder = 0)
        dots = ax.scatter([], [], [], c="#ff0000", zorder=5, label="Population")
        prime = ax.scatter([], [], [], s=75, c="#ffff00", zorder=10, label="Best Individual")

        ax.legend()
        ax.grid(True)

        def update(frame):
            self.select()
            fig.suptitle("Genetic" + str(frame + 1) + "/" + str(iterations) + " Best: " + str(
                round(self.func(self.pop[0]), 12)))
            x_coords = [p[0] for p in self.pop]
            y_coords = [p[1] for p in self.pop]
            z_coords = [self.func(p) for p in self.pop]

            dots._offsets3d = (x_coords, y_coords, z_coords)
            prime._offsets3d = ([self.pop[0][0]],
                                [self.pop[0][1]],
                                [self.func(self.pop[0])])
            self.crossover()
            self.mutate()
            if frame >= iterations - 1:
                ani.pause()
            return dots, prime
        if save:
            writervideo = animation.PillowWriter(fps=5, bitrate=1800)
            ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=100)
            ani.save("gifs/genetic_latest.gif", writer = writervideo)
        else:
            ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=100)
            plt.show()