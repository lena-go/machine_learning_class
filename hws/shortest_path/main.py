from graph_maker import CustomGraph
from graph_definer import random_edges
from algorithm import build_shortest_path, remove_longest_edges


def run(n: int = 10) -> None:
    graph = CustomGraph(n, random_edges(n))
    graph.draw(0, 'Initial graph')

    tree = build_shortest_path(n, graph.get_weights())
    graph.update_edges(tree)
    graph.draw(1, 'Tree')

    remaining_edges = remove_longest_edges(graph.get_weights(), k=3)
    graph.update_edges(remaining_edges)
    graph.draw(2, 'Clusters')

    clusters = graph.define_clusters(n)
    print('Clusters:')
    print(clusters)

    graph.show()


if __name__ == '__main__':
    run(10)
