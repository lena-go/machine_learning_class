import random
import networkx as nx
import matplotlib.pyplot as plt


class CustomGraph:
    def __init__(self, n: int, edges: [(int, int, int)]) -> None:
        self.G = nx.Graph()
        self.dots = list(range(n))
        self.G.add_nodes_from(self.dots)
        self.G.add_weighted_edges_from(edges)
        self.pos = nx.spring_layout(self.G)

    def draw(self, fig: int, plot_title: str = '') -> None:
        plt.figure(fig)
        plt.title(plot_title)
        weights = self.get_weights()
        color_options = {"edge_color": "tab:blue", "node_color": "tab:blue"}
        nx.draw(self.G, pos=self.pos, with_labels=True, **color_options)
        nx.draw_networkx_edge_labels(self.G, pos=self.pos, edge_labels=weights)

    def define_clusters(self, n: int) -> {int: [int]}:
        groups = {}
        last_group_idx = 0
        not_assigned = set(range(n))
        while not_assigned:
            dot = not_assigned.pop()
            groups[last_group_idx] = [dot]
            neighbours_to_check = list(self.G.neighbors(dot))
            for neighbour in neighbours_to_check:
                if neighbour in not_assigned:
                    groups[last_group_idx].append(neighbour)
                    not_assigned.remove(neighbour)
                    neighbours_to_check += self.G.neighbors(neighbour)

            last_group_idx += 1

        return groups

    def get_weights(self) -> {(int, int): int}:
        return nx.get_edge_attributes(self.G, 'weight')

    def update_edges(self, edges: [(int, int, int)]) -> None:
        self.G = nx.Graph()
        self.G.add_nodes_from(self.dots)
        self.G.add_weighted_edges_from(edges)

    def generate_color_options(self, clusters_count: int) -> [{str: str}]:
        options = []
        for cluster in range(clusters_count):
            color = self.generate_color()
            options.append({'edge_color': color, 'node_color': color})
        return options

    @staticmethod
    def generate_color() -> str:
        return '#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

    @staticmethod
    def show() -> None:
        plt.show()
