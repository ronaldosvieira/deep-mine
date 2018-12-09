# -*- coding: utf-8 -*-
import numpy as np
from operator import methodcaller, attrgetter

class Individual:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness = None

    def evaluate():
        pass

class GeneticAlgorithm:
    def __init__(self, ind_gen, select, crossover, mutate):
        self.ind_gen = ind_gen
        self.select = select
        self.crossover = crossover
        self.mutate = mutate

    def run(self, **params):
        np.random.seed(params['seed'] if 'seed' in params else None)

        generation = 0

        info = []

        population = [self.ind_gen() for _ in range(params['N'])]

        while generation < params['G']:
            for ind in population:
                ind.evaluate()

            info.append(population)

            new_population = []

            elite = sorted(population, reverse = True, key = attrgetter('fitness'))
            elite = elite[:params['elitism']]

            if len(elite) > 0:
                new_population.extend(elite)

            while len(new_population) < params['N']:
                chosen_ind = self.select(population, 1)[0]

                new_population.append(self.mutate(chosen_ind))

            population = new_population
            generation += 1

        for ind in population:
            ind.evaluate()

        return sorted(population, reverse = True, key = attrgetter('fitness'))[0], info