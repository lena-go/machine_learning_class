import random
from random import randint, randrange, shuffle, choice

from equation import Equation


class Mating:
    def choose_way(self, candidates: [(int,)], equation: Equation):
        way = randint(0, 2)
        match way:
            case 0:
                print('panmixia')
                return self.panmixia(candidates)
            case 1:
                print('inbreeding')
                return self.inbreeding(candidates, equation)
            case 2:
                print('out-breeding')
                return self.outbreeding(candidates, equation)

    @staticmethod
    def panmixia(candidates: [(int,)]) -> [[(int,)]]:
        candidates_cp = candidates[:]
        shuffle(candidates_cp)
        mates = []
        while len(candidates_cp) > 0:
            if len(candidates_cp) >= 2:
                mates.append([candidates_cp.pop(), candidates_cp.pop()])
            else:

                mates.append([
                    candidates_cp.pop(),
                    mates[randrange(0, len(mates))][randrange(0, 2)]
                ])
        return mates

    @staticmethod
    def __non_panmixia(candidates: [(int,)], equation: Equation, inbreeding: bool) -> [[(int,)]]:
        fitness = {i: equation.calc_fitness(candidates[i]) for i in range(len(candidates))}
        fitness_cp = fitness.copy()
        mates = []

        extremum = float('+inf') if inbreeding else float('-inf')
        extremum_indices = [None, None]
        while len(fitness) > 0:
            if len(fitness) >= 2:
                for k1, v1 in fitness.items():
                    for k2, v2 in fitness.items():
                        if k2 > k1:
                            dif = abs(v2 - v1)
                            if (
                                    (inbreeding and dif < extremum)
                                    or (not inbreeding and dif > extremum)
                            ):
                                extremum = dif
                                extremum_indices = [k1, k2]
                mates.append([
                    candidates[extremum_indices[0]], candidates[extremum_indices[1]]
                ])
                del fitness[extremum_indices[0]], fitness[extremum_indices[1]]
                extremum = float('+inf') if inbreeding else float('-inf')
                extremum_indices = [None, None]
            else:
                last_el_ind = list(fitness.keys())[0]
                extremum_ind = None
                for k, v in fitness_cp.items():
                    if k != last_el_ind:
                        dif = abs(v - fitness[last_el_ind])
                        if (
                                (inbreeding and dif < extremum)
                                or (not inbreeding and dif > extremum)
                        ):
                            extremum = dif
                            extremum_ind = k
                mates.append([candidates[last_el_ind], candidates[extremum_ind]])
                del fitness[last_el_ind]

        return mates

    def inbreeding(self, candidates: [(int,)], equation: Equation) -> [[(int,)]]:
        return self.__non_panmixia(candidates, equation, inbreeding=True)

    def outbreeding(self, candidates: [(int,)], equation: Equation) -> [[(int,)]]:
        return self.__non_panmixia(candidates, equation, inbreeding=False)


class GeneticAlgorithm:
    def __init__(
            self,
            equation: Equation,
            mutation_rate: float,
            survivors_ratio: float,
            max_generations: int,
            natural: bool = True,
    ) -> None:
        self.equation = equation
        self.population = [
            tuple(randint(1, self.equation.upper_borders[i]) for i in range(equation.terms_count))
            for _ in range(3)
        ]
        self.mating = Mating()
        self.mutation_rate = mutation_rate
        self.survivors_ratio = survivors_ratio
        self.max_generations = max_generations
        self.min_population_size = 8
        self.nearest_critter = None
        self.best_fitness = float('+inf')

    @staticmethod
    def crossover_couple(mother: (int,), father: (int,)) -> (int,):
        child = []
        for i in range(len(mother)):
            if i % 2 == 0:
                gene = mother[i]
            else:
                gene = father[i]
            child.append(gene)
        return tuple(child)

    def crossover(self, mates: [[(int,)]]) -> [(int,)]:
        children = []
        for couple in mates:
            children.append(self.crossover_couple(couple[0], couple[1]))
        return children

    def mutate_critters(self, critters: [(int,)]) -> [(int,)]:
        target_mutations_count = int(self.mutation_rate * len(critters))
        critters_upd = []
        shuffle(critters)
        for i in range(target_mutations_count):
            critters_upd.append(self.mutate_critter(critters[i]))
        critters_upd.extend(critters[target_mutations_count:])
        return critters_upd

    def mutate_critter(self, critter: (int,)) -> (int,):
        mutation_site = randrange(0, len(critter))
        new_critter = []
        for i in range(len(critter)):
            if i == mutation_site:
                gene = randint(1, self.equation.upper_borders[i])
            else:
                gene = critter[i]
            new_critter.append(gene)
        return tuple(new_critter)

    def select(self) -> None:
        fitness = [
            self.equation.calc_fitness(self.population[i])
            for i in range(len(self.population))
        ]
        target_survivors_count = self.survivors_ratio * len(self.population)
        while len(self.population) > target_survivors_count:
            shoot = randrange(sum(fitness))
            ind_of_dead_one = self.find_dead_one(shoot, fitness)
            del fitness[ind_of_dead_one]
            del self.population[ind_of_dead_one]

    @staticmethod
    def find_dead_one(shoot: int, fitness: {int: int}) -> int:
        i = 0
        critter_ind = 0
        while i <= shoot:
            i += fitness[critter_ind]
            critter_ind += 1
        return critter_ind - 1

    def find_nearest_answer(self, critters: [(int,)]) -> None:
        for critter in critters:
            fitness = self.equation.calc_fitness(critter)
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.nearest_critter = critter

    def run(self):
        generation_count = 0
        self.find_nearest_answer(self.population)
        while self.best_fitness != 0 and generation_count < self.max_generations:
            print(f"{'=' * 20} generation #{generation_count} {'=' * 20}")
            print('population:', self.population)
            while len(self.population) < self.min_population_size:
                mates = self.mating.choose_way(self.population, self.equation)
                print('mates:', mates)
                children = self.crossover(mates)
                print('children:', children)
                self.find_nearest_answer(children)
                mutated_children = self.mutate_critters(children)
                print('children after mutations:', mutated_children)
                self.population.extend(mutated_children)
                print('total population:', self.population)
                self.find_nearest_answer(mutated_children)
                generation_count += 1
            self.select()
            print('survivors:', self.population)
        return self.nearest_critter, self.best_fitness
