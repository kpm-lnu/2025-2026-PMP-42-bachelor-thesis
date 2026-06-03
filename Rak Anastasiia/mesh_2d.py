import numpy as np
import base_functions_2d as b2f


def uniform_mesh_with_vertices(vertices, p, m, element_type, degree=1):
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    x3, y3 = vertices[3]

    nodes_x = degree * p + 1
    nodes_y = degree * m + 1

    NL = np.zeros([nodes_x * nodes_y, 2])

    if degree == 1:
        x_coords = np.array([x0, x1, x2, x3])
        y_coords = np.array([y0, y1, y2, y3])
    elif degree == 2:
        x4, y4 = (x1 + x0) / 2, (y1 + y0) / 2
        x5, y5 = (x1 + x2) / 2, (y1 + y2) / 2
        x6, y6 = (x3 + x2) / 2, (y3 + y2) / 2
        x7, y7 = (x3 + x0) / 2, (y3 + y0) / 2
        x8, y8 = (x1 + x2 + x3 + x0) / 4, (y1 + y2 + y3 + y0) / 4
        x_coords = np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8])
        y_coords = np.array([y0, y1, y2, y3, y4, y5, y6, y7, y8])
    elif degree == 3:
        x4, y4 = (2 * x0 + x1) / 3, (2 * y0 + y1) / 3
        x5, y5 = (x0 + 2 * x1) / 3, (y0 + 2 * y1) / 3
        x6, y6 = (2 * x1 + x2) / 3, (2 * y1 + y2) / 3
        x7, y7 = (x1 + 2 * x2) / 3, (y1 + 2 * y2) / 3
        x8, y8 = (2 * x2 + x3) / 3, (2 * y2 + y3) / 3
        x9, y9 = (x2 + 2 * x3) / 3, (y2 + 2 * y3) / 3
        x10, y10 = (2 * x3 + x0) / 3, (2 * y3 + y0) / 3
        x11, y11 = (x3 + 2 * x0) / 3, (y3 + 2 * y0) / 3
        x12, y12 = (2 * x11 + x6) / 3, (2 * y11 + y6) / 3
        x13, y13 = (x11 + 2 * x6) / 3, (y11 + 2 * y6) / 3
        x14, y14 = (x10 + 2 * x7) / 3, (y10 + 2 * y7) / 3
        x15, y15 = (2 * x10 + x7) / 3, (2 * y10 + y7) / 3
        x_coords = np.array([x0, x1, x2, x3, x4, x6, x8, x10, x5, x7, x9, x11, x12, x13, x14, x15])
        y_coords = np.array([y0, y1, y2, y3, y4, y6, y8, y10, y5, y7, y9, y11, y12, y13, y14, y15])

    for i in range(nodes_y):
        for j in range(nodes_x):
            ksi = 2 * j / (nodes_x - 1) - 1
            eta = 2 * i / (nodes_y - 1) - 1

            x, y = isoparametric_transform(ksi, eta, x_coords, y_coords, degree)
            NL[i * nodes_x + j, :] = [x, y]

    if degree == 1:
        NPE = 4
    elif degree == 2:
        NPE = 9
    elif degree == 3:
        NPE = 16
    else:
        raise ValueError("Параметр degree повинен бути 1, 2 або 3.")

    NoE = p * m
    EL = np.zeros([NoE, NPE], dtype=int)

    e = 0
    for i in range(m):
        for j in range(p):
            base = i * degree * nodes_x + j * degree
            if degree == 1:
                # 4 вузли на елемент
                n1 = base
                n2 = n1 + 1
                n3 = n1 + nodes_x
                n4 = n3 + 1
                EL[e, :] = [n1, n2, n4, n3]
            elif degree == 2:
                # 9 вузлів на елемент
                n1 = base  # 1
                n2 = n1 + 2  # 2
                n3 = n1 + 2 * nodes_x  # 4
                n4 = n3 + 2  # 3
                n5 = n1 + 1  # 5
                n6 = n3 + 1  # 7
                n7 = n1 + nodes_x  # 8
                n8 = n7 + 2  # 6
                n9 = n7 + 1  # 9
                EL[e, :] = [n1, n2, n4, n3, n5, n8, n6, n7, n9]
            elif degree == 3:
                # 16 вузлів на елемент
                n1 = base  # 1
                n2 = n1 + 3  # 2
                n3 = n1 + 3 * nodes_x  # 4
                n4 = n3 + 3  # 3
                n5 = n1 + 1  # 5
                n6 = n1 + 2  # 9
                n7 = n3 + 1  # 11
                n8 = n3 + 2  # 7
                n9 = n1 + nodes_x  # 12
                n10 = n9 + 1  # 13
                n11 = n9 + 2  # 14
                n12 = n9 + 3  # 6
                n13 = n1 + 2 * nodes_x  # 8
                n14 = n13 + 1  # 16
                n15 = n13 + 2  # 15
                n16 = n13 + 3  # 10
                EL[e, :] = [n1, n2, n4, n3, n5, n12, n8, n13, n6, n16, n7, n9, n10, n11, n15, n14]

            e += 1

    return NL, EL


def isoparametric_transform(ksi, eta, x_coords, y_coords, ap):
    num_nodes = 4 if ap == 1 else (9 if ap == 2 else 16)
    x = sum(b2f.N(i, ksi, eta, ap) * x_coords[i] for i in range(num_nodes))
    y = sum(b2f.N(i, ksi, eta, ap) * y_coords[i] for i in range(num_nodes))
    return x, y