def build_shortest_path(n: int, weights: {(int, int): int}) -> [(int, int, int)]:
    if n < 2:
        return []

    smallest_weight_key = min(weights, key=weights.get)
    tree = [
        (smallest_weight_key[0],
         smallest_weight_key[1],
         weights[smallest_weight_key])
    ]

    non_isolated_ps = {smallest_weight_key[0], smallest_weight_key[1]}
    isolated_ps = {i for i in range(n) if i not in non_isolated_ps}

    while isolated_ps:

        smallest_weight = {
            'isolated': None,
            'non-isolated': None,
            'weight': float('+inf')
        }

        for non_isolated_point in non_isolated_ps:
            for isolated_point in isolated_ps:
                key = None
                if (non_isolated_point, isolated_point) in weights.keys():
                    key = (non_isolated_point, isolated_point)
                elif (isolated_point, non_isolated_point) in weights.keys():
                    key = (isolated_point, non_isolated_point)
                if key is not None and weights[key] < smallest_weight['weight']:
                    smallest_weight = {
                        'isolated': isolated_point,
                        'non-isolated': non_isolated_point,
                        'weight': weights[key]
                    }

        if smallest_weight['isolated'] is not None:
            isolated_ps.remove(smallest_weight['isolated'])
            non_isolated_ps.add(smallest_weight['isolated'])

            tree.append(
                (smallest_weight['isolated'],
                 smallest_weight['non-isolated'],
                 smallest_weight['weight'])
            )

    return tree


def remove_longest_edges(weights: {(int, int): int}, k: int = 3) -> [(int, int, int)]:
    sorted_edges = [(e1, e2, w) for (e1, e2), w in sorted(weights.items(), key=lambda item: item[1])]
    if k <= len(sorted_edges):
        for edge in range(k - 1):
            try:
                del sorted_edges[-1]
            except IndexError:
                pass
    return sorted_edges
