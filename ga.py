import numpy as np
import os.path
import time
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

    def init_population(self, params):
        if 'file' in params and os.path.exists(params['file']):
            info = list(map(list, np.load(params['load'])))
            
            population = info[-1]
        else:
            info = []

            population = [self.ind_gen() for _ in range(params['N'])]

        return population, info

    def run_generation(self, population, params):
        for i, ind in enumerate(population):
            if ind.fitness == None:
                print('########### EVALUATING {}/{} ###########'.format(i + 1, params['N']))
                ind.fitness = ind.evaluate(params.get('eval', 1))
                print(ind, "got", ind.fitness, "fitness!")

        new_population = []

        elite = self.sort_pop_by_fitness(population)

        if 'top_eval' in params:
            for i, ind in enumerate(population[:params['top_eval'][0]]):
                print('########### EVALUATING TOP {}/{} ###########' \
                    .format(i + 1, params['top_eval'][0]))
                ind.fitness = ind.evaluate(params['top_eval'][1])
                print(ind, "got", ind.fitness, "fitness!")

        elite = elite[:params['elitism']]

        if len(elite) > 0:
            new_population.extend(elite)

        while len(new_population) < params['N']:
            chosen_ind = self.select(population, 1)[0]

            new_population.append(self.mutate(chosen_ind))

        for i, ind in enumerate(new_population):
            print('########### EVALUATING {}/{} ###########'.format(i + 1, params['N']))
            ind.fitness = ind.evaluate(1)
            print(ind, "got", ind.fitness, "fitness!")

        return new_population

    def sort_pop_by_fitness(self, population):
        return sorted(population, reverse = True, key = attrgetter('fitness'))

    def run(self, **params):
        np.random.seed(params['seed'] if 'seed' in params else None)
        params['file'] = params.get('file', 
                    'runs/{}.npy'.format(time.strftime("%Y%m%d-%H%M%S")))

        population, info = self.init_population(params)

        generation = 1

        while generation <= params['G']:
            print('########### GENERATION {} ###########'.format(generation))

            new_population = self.run_generation(population, params)

            info.append(self.sort_pop_by_fitness(population))
            np.save(params['file'], info)

            population = new_population
            generation += 1

        info.append(self.sort_pop_by_fitness(population))
        np.save(params['file'], info)

        return self.sort_pop_by_fitness(population)[0], info