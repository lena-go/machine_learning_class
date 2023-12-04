from random import randint
from math import ceil
from string import ascii_lowercase
from typing import List


ANSWER_BORDERS = [-100, 100]
COEF_BORDERS = [-20, 20]


class Equation:
    def __init__(
            self,
            terms_count: int | None = None,
            coefficients: List[int] | None = None,
            right_part: int | None = None,
    ) -> None:
        if isinstance(coefficients, list):
            self.coefficients = coefficients
            self.terms_count = len(coefficients)
        elif isinstance(terms_count, int) and terms_count > 0:
            self.terms_count = terms_count
            self.coefficients = []
            for _ in range(terms_count):
                c = 0
                while c == 0:
                    c = randint(-10, 10)
                self.coefficients.append(c)

            # self.coefficients = [randint(-20, 20) for _ in range(terms_count)]
        else:
            raise ValueError('Incorrect type or value of parameter "coefficients" or parameter "terms_count"')

        if right_part is not None:
            self.right_part = right_part
        else:
            min_sum = sum(self.coefficients)
            # min_sum = 0
            self.right_part = randint(min_sum, min_sum + 1000)

        self.upper_borders = [ceil(self.right_part / c) for c in self.coefficients]
        print(self.upper_borders)

    def evaluate(self, args: [int]) -> int:
        if len(self.coefficients) != len(args):
            raise ValueError('Different number of args and coefficients in the equation')

        sm = 0
        for i in range(len(args)):
            sm += self.coefficients[i] * args[i]

        return sm

    def calc_fitness(self, args: [int]) -> int:
        return abs(self.evaluate(args) - self.right_part)

    def __str__(self):
        left_part = []
        if self.terms_count <= len(ascii_lowercase):
            for i in range(self.terms_count):
                left_part.append(str(self.coefficients[i]) + ascii_lowercase[i])
        else:
            for i in range(self.terms_count):
                arg = 'x_' + str(i)
                left_part.append(str(self.coefficients[i]) + arg)
        left_part_str = ' + '.join(left_part)
        return left_part_str + ' = ' + str(self.right_part)
