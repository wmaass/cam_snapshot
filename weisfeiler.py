import networkx as nx

G1 = nx.Graph()
G1.add_edges_from(
    [
        (1, 2, {"label": "A"}),
        (2, 3, {"label": "A"}),
        (3, 1, {"label": "A"}),
        (1, 4, {"label": "B"}),
    ]
)
G2 = nx.Graph()
G2.add_edges_from(
    [
        (5, 6, {"label": "B"}),
        (6, 7, {"label": "A"}),
        (7, 5, {"label": "A"}),
        (7, 8, {"label": "A"}),
    ]
)

# visualize graphs
import matplotlib.pyplot as plt
plt.subplot(121)
nx.draw(G1, with_labels=True, font_weight='bold')
plt.subplot(122)
nx.draw(G2, with_labels=True, font_weight='bold')
plt.show()

hash1 = nx.weisfeiler_lehman_graph_hash(G1)
hash2 = nx.weisfeiler_lehman_graph_hash(G2)

if hash1 == hash2:
    print("The graphs might be isomorphic.")
else:
    print("The graphs are not isomorphic.")
