from random import randint
from string import ascii_lowercase
from typing import List


class Equation:
    def __init__(
            self,
            terms_count: int | None = None,
            coefficients: List[int] | None = None,
            right_part: int | None = None,
            answer_borders: (int, int) = (-100, 100),
            coef_borders: (int, int) = (-10, 10),
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
                    c = randint(*coef_borders)
                self.coefficients.append(c)
        else:
            raise ValueError('Incorrect type or value of parameter "coefficients" or parameter "terms_count"')

        if right_part is not None:
            self.right_part = right_part
        else:
            min_sum = 0
            for c in self.coefficients:
                if c < 0:
                    min_sum += c * answer_borders[1]
                elif c > 0:
                    min_sum += c * answer_borders[0]
            max_sum = -min_sum
            print('min_sum', min_sum)
            print('max_sum', max_sum)
            self.right_part = randint(min_sum, max_sum)

        self.answer_borders = answer_borders

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
