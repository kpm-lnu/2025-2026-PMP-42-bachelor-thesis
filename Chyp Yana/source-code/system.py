# assembles system (matrix) of elements
# independently of their kind
import elasticdeform as elde
import local
from elasticdeform import ElasticDeform
from local import LTriangle



class System:
    elde: ElasticDeform
    def __init__(self, elde):
        self.elde = elde
    def assemble_matrix(self, elements, k_e, nn):
        # nn - number of nodes
        matrix = [[0 for j in range(2*nn)] for i in range(2*nn)]
        for element in elements:
            m = len(element.nodes)
            for i in range(m):
                ion = element.nodes[i]
                for j in range(m):
                    jon = element.nodes[j]
                    matrix[2*ion][2*jon] += element.jacobian * k_e[element.approx-1][2*i][2*j]
                    matrix[2*ion+1][2*jon+1] += element.jacobian * k_e[element.approx-1][2*i+1][2*j+1]
        return matrix

    def assemble_matrix_physical(self, elements, k_e_phys, nn):
        # k_e_phys: (k_e1_list, k_e2_list, k_e3_list) — per-element matrices from stiffness_matrix_physical
        matrix = [[0 for j in range(2*nn)] for i in range(2*nn)]
        counters = [0, 0, 0]
        for element in elements:
            m = len(element.nodes)
            idx = counters[element.approx - 1]
            k_e = k_e_phys[element.approx - 1][idx]
            counters[element.approx - 1] += 1
            for i in range(m):
                ion = element.nodes[i]
                for j in range(m):
                    jon = element.nodes[j]
                    matrix[2*ion][2*jon]     += k_e[2*i][2*j]
                    matrix[2*ion+1][2*jon+1] += k_e[2*i+1][2*j+1]
                    matrix[2*ion][2*jon+1]   += k_e[2*i][2*j+1]
                    matrix[2*ion+1][2*jon]   += k_e[2*i+1][2*j]
        return matrix

    def assemble_vector(self, elements, nn, b):
        # elde = ElasticDeform(base)
        vector = [0 for i in range(2*nn)]
        for element in elements:
            m = len(element.nodes)
            r_e = self.elde.load_vector(b, element)
            for i in range(m):
                ion = element.nodes[i]
                vector[2*ion] += element.jacobian / 2 * r_e[element.approx-1][2*i]
                vector[2*ion+1] += element.jacobian / 2 * r_e[element.approx-1][2*i+1]
        return vector


# ed = elde.ElasticDeform(base=LTriangle.base())
# k_e = ed.stiffness_matrix()
# print(k_e)
# is [ 6x6 ] for 3-node triangle

