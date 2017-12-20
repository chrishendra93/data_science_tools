def convert_mst_to_edge_list(mst):
    return {(edge[1], edge[2]) for edge in mst}


def convert_edge_list_to_dict(edge_list, features):
    edge_dict = {feature: set([]) for feature in features}
    for edge in edge_list:
        edge_dict[edge[0]].add(edge[1])
        edge_dict[edge[1]].add(edge[0])
    return edge_dict


def create_dependencies_dict(graph_dict, features, label_var):
    def convert_graph_dict_to_dag(graph_dict, visited_vertices, dag, vertex):
        if vertex not in visited_vertices:
            visited_vertices.append(vertex)
            for child in graph_dict[vertex]:
                if child not in visited_vertices:
                    if vertex not in dag:
                        dag[vertex] = set([child])
                    else:
                        dag[vertex].add(child)
                    convert_graph_dict_to_dag(graph_dict, visited_vertices, dag, child)
    dag = {}
    root = graph_dict.keys()[0]
    convert_graph_dict_to_dag(graph_dict, [], dag, root)
    parents_list = {feature: [] for feature in features}
    for par, children in dag.iteritems():
        for child in children:
            parents_list[child].append(par)
    return parents_list
