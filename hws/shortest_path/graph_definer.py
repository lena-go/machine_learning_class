import random


MAX_EDGES_COUNT_PER_VERTEX = 4


def check_if_has_adjacent_v(vertex: int, edges: [(int, int, int)]) -> bool:
    has_adjacent_v = False
    for edge in edges:
        if edge[1] == vertex:
            has_adjacent_v = True
            break
    return has_adjacent_v


def random_edges(n: int) -> [(int, int, int)]:
    if n < 2:
        return []

    edges = []

    for i in range(n):
        buffer = list(range(i + 1, n))

        edges_count = random.randint(0, min(n - i - 1, MAX_EDGES_COUNT_PER_VERTEX))
        if edges_count == 0 and not check_if_has_adjacent_v(i, edges):
            edges_count += 1
            if i == n - 1:
                buffer = list(range(n - 1))

        random.shuffle(buffer)

        for j in range(0, edges_count):
            weight = random.randint(1, 9)
            edges.append((i, buffer.pop(), weight))

    return edges
