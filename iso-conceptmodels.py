import os
import shutil
import re
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import isomorphism

class FileAnalyzer:
    def __init__(self, app_name, base_path):
        self.app_name = app_name
        self.base_path = base_path
        self.src_folder = os.path.join(base_path, app_name)
        self.final_folder = os.path.join(self.src_folder, 'final')
        self.pattern = re.compile(r'GCBM-' + app_name + '(\d+)-(\d+)')

    def copy_files_with_highest_y(self):
        highest_y_files = {}

        for file in os.listdir(self.src_folder):
            match = self.pattern.match(file)
            if match:
                x, y = map(int, match.groups())
                if x not in highest_y_files or y > highest_y_files[x][1]:
                    highest_y_files[x] = (file, y)

        final_folder = os.path.join(self.src_folder, 'final')
        os.makedirs(final_folder, exist_ok=True)

        for file, _ in highest_y_files.values():
            shutil.copy(os.path.join(self.src_folder, file), os.path.join(final_folder, file))

    # create a function that compares all graphml files in folder 'final' and returs the names of all graphs that are isomorphic to each other
    def find_isomorphic_graphs(self):
        graph_files = [f for f in os.listdir(self.final_folder) if f.endswith('.graphml')]
        graphs = {f: nx.read_graphml(os.path.join(self.final_folder, f)) for f in graph_files}
        
        isomorphic_pairs = []

        for i in range(len(graph_files)):
            for j in range(i + 1, len(graph_files)):
                file1 = graph_files[i]
                file2 = graph_files[j]

                if isomorphism.is_isomorphic(graphs[file1], graphs[file2]):
                    isomorphic_pairs.append((file1, file2))

        return isomorphic_pairs
    
    def plot_graph_from_pairs(self, pairs):
        G = nx.Graph()
        G.add_edges_from(pairs)

        plt.figure(figsize=(10, 8))
        nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        plt.show()

    # create a function that calculates the average number of conncects per node in the isomorphic_pairs list
    def average_connects(self, pairs):
        G = nx.Graph()
        G.add_edges_from(pairs)
        connects = nx.average_node_connectivity(G)
        return connects

# Usage
app_name = "HIS17-onehot-two-classes"
path_graphs = '/Users/wolfgangmaass/Documents/Github/ConceptualAlignment/graphs'

organizer = FileAnalyzer(app_name, path_graphs)

organizer.copy_files_with_highest_y()

isomorphic_graphs = organizer.find_isomorphic_graphs()

# create a graph from isomorphic_graphs and plot it
organizer.plot_graph_from_pairs(isomorphic_graphs)

print(organizer.average_connects(isomorphic_graphs))
