import networkx as nx
from itertools import combinations
import gravis as gv
import os

class GraphUtil:
    def build_graph(self, tokens):
        G = nx.DiGraph()
        G.add_nodes_from(tokens)
        for i in range(len(tokens) - 1):
            G.add_edge(tokens[i], tokens[i + 1])
        return G
    
    def get_score(self, g1, g2):
        matching_graph = nx.Graph()
        for n1, n2 in g2.edges():
            if g1.has_edge(n1, n2):
                matching_graph.add_edge(n1, n2)
        components = list(nx.connected_components(matching_graph))
        return sum([len(i) for i in components]) / min(g1.number_of_nodes(), g2.number_of_nodes())

    def save_graph(self, G, folder, file):
        if not os.path.exists(folder):
            os.makedirs(folder)
        graph_string = ""
        graph_string += "t # 0\n"
        for node, data in G.nodes(data=True):
            graph_string += f"v {node} 1\n"
        for u, v, data in G.edges(data=True):
            graph_string += f"e {u} {v} 1\n"
        graph_string += "t # -1\n"
        with open(os.path.join(folder, file), "w", encoding='utf-8') as file:
            file.write(graph_string)

    def get_mcs(self, g1, g2):
        matching_graph = nx.Graph()
        for n1, n2 in g2.edges():
            if g1.has_edge(n1, n2):
                matching_graph.add_edge(n1, n2)
        components = list(nx.connected_components(matching_graph))
        if components:
            largest_component = max(components, key=len)
            return nx.induced_subgraph(matching_graph, largest_component)
        else:
            return nx.Graph()

    def find_subgraphs(self, graphs, min_support):
        subgraphs = []
        for graph in graphs:
            for node in graph.nodes():
                for k in range(min_support, len(graph.nodes()) + 1):
                    for subset in combinations(graph.nodes(), k):
                        subgraph = graph.subgraph(subset)
                        if nx.is_weakly_connected(subgraph):
                            subgraphs.append(subgraph)
        return subgraphs
    
    def visualize_graph(self, G):
        gv.d3(G).display()