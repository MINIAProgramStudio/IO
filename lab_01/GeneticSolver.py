from random import random
from tqdm import tqdm

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

    def select(self):
        self.pop = sorted(self.pop, key = self.func, reverse = not self.seeking_min) # відсортувати хромосоми за фітнес-функцією
        self.pop = self.pop[:self.pop_size] # залишити лише найкращі хромосоми у кількості pop_size штук
        return self.pop[0] # повернути найкращу хромосому

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
        self.select()
        self.crossover()
        self.mutate()

    def solve(self, iterations, progressbar = True, epsilon_timeout = float("inf"), epsilon = 0):
        if progressbar:
            iterator = tqdm(range(iterations))
        else:
            iterator = range(iterations)
        epsilon_timeout_counter = 0

        for _ in iterator:
            old_best = self.pop[0]
            self.iter()

            if self.func(self.pop[0]) - self.func(old_best) >= epsilon * (-1) ** self.seeking_min:
                epsilon_timeout_counter = 0
            else:
                epsilon_timeout_counter += 1
                if epsilon_timeout_counter > epsilon_timeout:
                    self.select()
                    return self.pop[0]
        self.select()
        return self.pop[0]