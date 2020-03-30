import numpy as np

def convert(tup):
    di = dict(tup)
    return di


def hadamard(model, data, link_info, protS_int):
    hadamard_links = []
    for i in range(len(data)):
        had_pro = np.multiply(model.wv[str(data[i][0])], model.wv[str(data[i][1])])
        absolute_diff = abs(protS_int[data[i][0]]-protS_int[data[i][1]])
        hadamard_links.append((had_pro, absolute_diff, link_info[i]))
    return hadamard_links


def get_tups_data(hadamard_data):
    vectors_and_abs_val = []
    links = []
    for i in hadamard_data:
        vectors_and_abs_val.append((i[0], i[1]))
        links.append(i[2])
    return vectors_and_abs_val, links


def transform_str_to_int(orig_node_list, edges):
    """node_list is the tuple list in order of graph"""
    for i in range(len(edges)):
        ind_first_ele = [y[0] for y in orig_node_list].index(edges[i][0])
        ind_sec_ele = [y[0] for y in orig_node_list].index(edges[i][1])
        edges[i] = (ind_first_ele, ind_sec_ele)
    return edges


