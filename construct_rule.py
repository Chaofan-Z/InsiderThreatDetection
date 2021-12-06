import numpy as np
import networkx as nx
from tqdm import tqdm

# 一个用户同天同一个host时序连接
def rule_1(activity_graph, daily_sequences_list, day_delta):
    # list{day -> map{host->activity}}
    host_activity = [None] * day_delta
    for daily_sequence in tqdm(daily_sequences_list):
        h_tuple = {}
        if daily_sequence:
            nodes = list(daily_sequence.nodes())
            for node_id in nodes:
                host = daily_sequence.nodes[node_id]['H']
                if host not in h_tuple:
                    h_tuple[host] = [node_id]
                else:
                    h_tuple[host].append(node_id)
                    daily_sequence.add_edge(h_tuple[host][-2], h_tuple[host][-1], EdgeType=1, weight=1)
            host_activity[daily_sequences_list.index(daily_sequence)] = h_tuple
    # activity_graph 接口形式统一，并未用到该变量
    return activity_graph, host_activity

# 一个用户多天同一个host的行为链的时序关联
def rule_2(activity_graph, daily_sequences_list, day_delta, host_activity):
    for daily_sequence in daily_sequences_list:
        if daily_sequence:
            activity_graph = nx.compose(activity_graph, daily_sequence)
    for day_i in range(day_delta):
        for day_j in range(day_i + 1, day_delta):
            if not (daily_sequences_list[day_i] and daily_sequences_list[day_j]) :
                continue

            for host in host_activity[day_i]:
                if host in host_activity[day_j]:

                    st_i = host_activity[day_i][host][0]
                    ed_i = host_activity[day_i][host][-1]

                    st_j = host_activity[day_j][host][0]
                    ed_j = host_activity[day_j][host][-1]

                    len_i = len(host_activity[day_i][host])
                    len_j = len(host_activity[day_j][host])

                    weight = len_i / len_j if len_i < len_j else len_j / len_i
                    activity_graph.add_edge(st_i, st_j, EdgeType=2, weight=weight)
                    activity_graph.add_edge(ed_i, ed_j, EdgeType=2, weight=weight)
    
    return activity_graph

# 一个用户多天同一个host同种组操作类型时序关联
# （规则定义组操作类型，比如Connect-> disconnect, File open -> File Write, visit web...）
def rule_3(activity_graph, daily_sequences_list, day_delta):
    return activity_graph

