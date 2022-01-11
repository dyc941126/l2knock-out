import random

import torch

from core.parser import parse
X_EMBED = [1, 0, 0, 0, 0]
F_EMBED = [0, 1, 0, 0, 0]
N_EMBED = [0, 0, 1, 0, 0]
C_EMBED = [0, 0, 0, 1]
ID_EMBED_UB = 0.1


class Environment:
    def __init__(self, pth, solver, scale=1, id_embed_dim=0):
        all_vars, self.all_functions = parse(pth, scale=scale)
        self.dom_size = dict()
        self.id_embed = dict()
        for name, dom_size in all_vars:
            self.dom_size[name] = dom_size
            id_embed = [] if id_embed_dim <= 0 else [random.random() * ID_EMBED_UB for _ in range(id_embed_dim)]
            self.id_embed[name] = id_embed
        if id_embed_dim > 0:
            for name, _ in all_vars:
                norm = sum(self.id_embed[name])
                self.id_embed[name] = [x / norm for x in self.id_embed[name]]

        self.solver = solver
        self.current_cost = self.solver(self.dom_size, self.all_functions)
        self.degenerated_vars = 0

    def act(self, name, dom_val):
        new_functions = []
        for func, var1, var2 in self.all_functions:
            new_func = []
            if var1 == name:
                assert dom_val < self.dom_size[var1] and self.dom_size[var1] > 1
                self.dom_size[var1] -= 1
                if self.dom_size[var1] == 1:
                    self.degenerated_vars += 1
                for i, row in enumerate(func):
                    if i != dom_val:
                        new_func.append(row)
            elif var2 == name:
                assert dom_val < self.dom_size[var2] and self.dom_size[var2] > 1
                self.dom_size[var2] -= 1
                if self.dom_size[var2] == 1:
                    self.degenerated_vars += 1
                for row in func:
                    new_func.append([row[i] for i in range(len(row)) if i != dom_val])
            else:
                new_func = func
            new_functions.append((new_func, var1, var2))
        self.all_functions = new_functions
        current_cost = self.solver(self.dom_size, self.all_functions)
        r = self.current_cost - current_cost
        self.current_cost = current_cost
        return r, self.degenerated_vars == len(self.dom_size)

    def observe(self):
        function_node_index = []
        assignment_node_start_index = dict()
        inverse_assignment_node_index = dict()
        x = []
        for var, dom_size in self.dom_size.items():
            assignment_node_start_index[var] = len(x)
            for i in range(dom_size):
                inverse_assignment_node_index[len(x)] = var
                x.append(X_EMBED + self.id_embed[var])
        x.append(N_EMBED)
        action_space_limit = len(x)
        edge_index = [[], []]
        src, dest = edge_index

        for func, var1, var2 in self.all_functions:
            f_idx = len(x)
            function_node_index.append(f_idx)
            x.append(F_EMBED)
            for i in range(self.dom_size[var1]):
                for j in range(self.dom_size[var2]):
                    c_idx = len(x)
                    x.append(C_EMBED + [func[i][j]])

                    src.append(c_idx)
                    dest.append(f_idx)

                    src.append(assignment_node_start_index[var1] + i)
                    dest.append(c_idx)
                    src.append(c_idx)
                    dest.append(assignment_node_start_index[var2] + j)

                    src.append(assignment_node_start_index[var2] + j)
                    dest.append(c_idx)
                    src.append(c_idx)
                    dest.append(assignment_node_start_index[var1] + i)
        for f_idx in function_node_index:
            src.append(f_idx)
            dest.append(action_space_limit - 1)
        x = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return x, edge_index, action_space_limit, function_node_index, inverse_assignment_node_index, assignment_node_start_index
