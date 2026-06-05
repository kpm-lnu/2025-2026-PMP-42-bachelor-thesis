import numpy as np
import matplotlib.pyplot as plt

def create_beam_mesh(L=1.0, H=0.2, nx=20, ny=5, q8=True):

    n_nodes_x = nx + 1
    n_nodes_y = ny + 1

    nodes = np.zeros((n_nodes_x * n_nodes_y, 2))

    for j in range(n_nodes_y):
        y = -H / 2 + j * H / ny
        for i in range(n_nodes_x):
            x = i * L / nx
            idx = j * n_nodes_x + i
            nodes[idx] = [x, y]

    elements = []

    if not q8:
        for j in range(ny):
            for i in range(nx):
                n1 = j * n_nodes_x + i
                n2 = j * n_nodes_x + i + 1
                n3 = (j + 1) * n_nodes_x + i + 1
                n4 = (j + 1) * n_nodes_x + i
                elements.append([n1, n2, n3, n4])
        return nodes, np.array(elements)

    # -------------------------
    # Q8 — interleaved ordering to match shape_functions_q8:
    #
    #   idx: 0        1         2        3        4        5        6        7
    #        C(-1,-1) M(0,-1)  C(1,-1)  M(1,0)  C(1,1)  M(0,1)  C(-1,1)  M(-1,0)
    #
    #   Visually:
    #       6 --- 5 --- 4
    #       |           |
    #       7           3
    #       |           |
    #       0 --- 1 --- 2
    # -------------------------

    node_map = {}
    nodes_list = nodes.tolist()

    def get_midpoint(nA, nB):
        edge = tuple(sorted((nA, nB)))
        if edge in node_map:
            return node_map[edge]
        midpoint = 0.5 * (nodes[nA] + nodes[nB])
        idx = len(nodes_list)
        nodes_list.append(midpoint.tolist())
        node_map[edge] = idx
        return idx

    for j in range(ny):
        for i in range(nx):
            n1 = j * n_nodes_x + i             # bottom-left  (-1,-1)
            n2 = j * n_nodes_x + i + 1         # bottom-right  (1,-1)
            n3 = (j + 1) * n_nodes_x + i + 1   # top-right     (1, 1)
            n4 = (j + 1) * n_nodes_x + i       # top-left     (-1, 1)

            n5 = get_midpoint(n1, n2)  # bottom mid  (0,-1)
            n6 = get_midpoint(n2, n3)  # right mid   (1, 0)
            n7 = get_midpoint(n3, n4)  # top mid     (0, 1)
            n8 = get_midpoint(n4, n1)  # left mid   (-1, 0)

            # Interleaved: corner, midside, corner, midside, ...
            elements.append([n1, n5, n2, n6, n3, n7, n4, n8])

    return np.array(nodes_list), np.array(elements)


def plot_mesh(nodes, elements, show_nodes=True, show_numbers=True, title="FEM Mesh"):

    plt.figure(figsize=(10, 3))

    for el in elements:
        el = np.array(el)

        if len(el) == 8:
            # corners are at even indices: 0, 2, 4, 6
            corners = [0, 2, 4, 6, 0]
            coords = nodes[el[corners]]
            plt.plot(coords[:, 0], coords[:, 1], 'k-')

            # dashed lines through midsides (odd indices)
            mids = [
                (0, 1), (1, 2),   # bottom edge
                (2, 3), (3, 4),   # right edge
                (4, 5), (5, 6),   # top edge
                (6, 7), (7, 0)    # left edge
            ]
            for a, b in mids:
                plt.plot(
                    [nodes[el[a], 0], nodes[el[b], 0]],
                    [nodes[el[a], 1], nodes[el[b], 1]],
                    'k--', linewidth=0.5
                )
        else:
            coords = nodes[el.tolist() + [el[0]]]
            plt.plot(coords[:, 0], coords[:, 1], 'k-')

    if show_nodes:
        plt.scatter(nodes[:, 0], nodes[:, 1], s=15, c='red', zorder=3)

    if show_numbers:
        for i, (x, y) in enumerate(nodes):
            plt.text(x, y, str(i), fontsize=6, color='blue', ha='center', va='center')

    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
