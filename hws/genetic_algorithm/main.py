from algorithm import GeneticAlgorithm
from equation import Equation


def run():
    # eq = Equation(coefficients=[3, 6, 15], right_part=120)
    eq = Equation(terms_count=10)
    # eq = Equation(coefficients=[2, 3], right_part=6)
    # alg = GeneticAlgorithm(
    #     eq,
    #     mutation_rate=0.5,
    #     survivors_ratio=0.5,
    #     max_generations=1000
    # )
    # ans = alg.run()
    # print('-' * 50)
    print(eq)
    # print('Answer:', ans)


if __name__ == '__main__':
    run()
