import numpy as np
import networkx as nx
from tqdm import tqdm

# 一个用户同天同一个host时序连接
def rule_1(activity_graph, daily_sequences_list, day_delta):
    print("同一用户同一天相同主机 Rule1 Starts!")

    H_tuple_list = [None] * day_delta

    for daily_sequence in daily_sequences_list:
        if daily_sequence:
            # key: H;    value: list of nodes number
            H_record_tuple = {}
            node_list = list(daily_sequence.nodes())
            for node_i in node_list:
                current_H = daily_sequence.nodes[node_i]['H']
                if current_H not in H_record_tuple.keys():
                    H_record_tuple[current_H] = [node_i]
                else:
                    # pdb.set_trace()
                    node_j = H_record_tuple[current_H][-1]
                    activity_graph.add_edge(node_j, node_i, EdgeType=1, weight=1)
                    H_record_tuple[current_H].append(node_i)

            day_of_seq = daily_sequences_list.index(daily_sequence)
            H_tuple_list[day_of_seq] = H_record_tuple

    print("同一用户同一天相同主机 Rule1 End!")
    # 这里先将多天要用的字典数据一次性先跑出来
    return activity_graph, H_tuple_list

# 一个用户多天同一个host的行为链的时序关联
def rule_2(activity_graph, daily_sequences_list, day_delta, H_tuple_list):
    print('同一用户多天同一主机 Rule2 Start！')

    # Add edges between daily sequences
    for i in range(0, day_delta):
        # 同一天
        for j in range(i + 1, day_delta):
            if daily_sequences_list[i] and daily_sequences_list[j]:
                # 同一台主机
                for key in H_tuple_list[i]:
                    if key in H_tuple_list[j].keys():
                        u1 = H_tuple_list[i][key][0]
                        v1 = H_tuple_list[j][key][0]
                        u2 = H_tuple_list[i][key][-1]
                        v2 = H_tuple_list[j][key][-1]
                        len_u = len(H_tuple_list[i][key])
                        len_v = len(H_tuple_list[j][key])
                        weight_u_v = len_u / len_v if len_u < len_v else len_v / len_u
                        w = round(weight_u_v, 3)
                        activity_graph.add_edge(u1, v1, EdgeType=2, weight=w)
                        activity_graph.add_edge(u2, v2, EdgeType=2, weight=w)
    print("同一用户多天同一主机 Rule2 End！")
    return activity_graph

# 一个用户多天同一个host同种组操作类型时序关联
# 这个需要先构建同一用户同一host一天的序列，然后将多天序列连起来
# （规则定义组操作类型，比如Connect-> disconnect, File open -> File Write, visit web...）
def rule_3(activity_graph, daily_sequences_list, day_delta, H_tuple_list):
    print('同一天同一主机相同操作类型 Rule3 Start！')

    print('需要先构建同一天同一主机相同操作类型 Start!')

    A_tuple_list = [None] * day_delta
    for daily_sequence in daily_sequences_list:
        if daily_sequence:
            # key: H;    value: list of nodes number
            day_of_seq = daily_sequences_list.index(daily_sequence)
            H_record_tuple = H_tuple_list[day_of_seq]


            A_record_tuple_tuple = {}
            for key in H_record_tuple:
                # Nodes in H_list have the same H
                H_list = H_record_tuple[key]
                A_record_tuple = {}
                for node_i in H_list:
                    current_A = daily_sequence.nodes[node_i]['A']
                    if current_A not in A_record_tuple.keys():
                        A_record_tuple[current_A] = [node_i]
                    else:
                        node_j = A_record_tuple[current_A][-1]
                        # 这里的边是同一用户同一天同一主机同一种操作类型
                        activity_graph.add_edge(node_j, node_i, EdgeType=3, weight=1)
                        A_record_tuple[current_A].append(node_i)

                A_record_tuple_tuple[key] = A_record_tuple

            A_tuple_list[day_of_seq] = A_record_tuple_tuple

    print('同一用户同一天同一主机同一种操作类型 End！')

    print('同一用户不同天同一主机相同操作类型 Start!')
    for i in range(0, day_delta):
        # 不同天之间
        for j in range(i + 1, day_delta):
            if daily_sequences_list[i] and daily_sequences_list[j]:
                # 同一台主机
                for key in H_tuple_list[i]:
                    if key in H_tuple_list[j].keys():
                        # 相同操作类型
                        for operation_type in A_tuple_list[i][key]:
                            if operation_type in A_tuple_list[j][key]:
                                u1 = A_tuple_list[i][key][operation_type][0]
                                v1 = A_tuple_list[j][key][operation_type][0]
                                u2 = A_tuple_list[i][key][operation_type][-1]
                                v2 = A_tuple_list[j][key][operation_type][-1]
                                len_u = len(A_tuple_list[i][key][operation_type])
                                len_v = len(A_tuple_list[j][key][operation_type])
                                weight_u_v = len_u / len_v if len_u < len_v else len_v / len_u
                                w = round(weight_u_v, 3)
                                activity_graph.add_edge(u1, v1, EdgeType=3, weight=w)
                                activity_graph.add_edge(u2, v2, EdgeType=3, weight=w)
    print('同一用户不同天同一主机相同操作类型 End!')
    print("同一天同一主机相同操作类型 Rule3 End!")
    return activity_graph, A_tuple_list

