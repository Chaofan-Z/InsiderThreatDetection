import networkx as nx
from tqdm import tqdm
from itertools import chain
import time

def timer(function):
    def wrapper(*args, **kwargs):
        time_start = time.time()
        print("%s start to construct" % (function.__name__))
        res = function(*args, **kwargs)
        cost_time = time.time() - time_start
        print("%s end to construct, cost %.2fs" % (function.__name__, cost_time))
        return res
    return wrapper


# 一个用户同天同一个host时序连接
@timer
def rule_1(activity_graph, daily_sequences_list, day_delta):
    # host_activity : list{day -> map{host->activity_list}}
    rule1_num = 0
    day_delta = len(daily_sequences_list)
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
                    rule1_num += 1
                    daily_sequence.add_edge(h_tuple[host][-2], h_tuple[host][-1], EdgeType=1, weight=1)
            host_activity[daily_sequences_list.index(daily_sequence)] = h_tuple
    print("rule1 edge number : ", rule1_num)
       
    # activity_graph 接口形式统一，并未用到该变量
    return activity_graph, host_activity

# 一个用户多天同一个host的行为链的时序关联
@timer
def rule_2(activity_graph, daily_sequences_list, day_delta, host_activity):
    day_delta = len(daily_sequences_list)
    rule2_num = 0
    for daily_sequence in daily_sequences_list:
        if daily_sequence:
            activity_graph = nx.compose(activity_graph, daily_sequence)
    for day_i in range(day_delta):
        for day_j in range(day_i + 1, day_delta):
            if not (daily_sequences_list[day_i] and daily_sequences_list[day_j]) :
                continue

            for host in host_activity[day_i]:
                if host in host_activity[day_j]:
                    # Todo : 可能某些序列中并不存在闭环的logon -> logoff，是否可以将logon logoff定为rule3的一个规则
                    st_i = host_activity[day_i][host][0]
                    ed_i = host_activity[day_i][host][-1]

                    st_j = host_activity[day_j][host][0]
                    ed_j = host_activity[day_j][host][-1]

                    len_i = len(host_activity[day_i][host])
                    len_j = len(host_activity[day_j][host])

                    weight = len_i / len_j if len_i < len_j else len_j / len_i
                    rule2_num += 2
                    activity_graph.add_edge(st_i, st_j, EdgeType=2, weight=weight)
                    activity_graph.add_edge(ed_i, ed_j, EdgeType=2, weight=weight)
    print("rule2 edge number : ", rule2_num)
    return activity_graph

# 一个用户同一个host下多天 同种组操作类型时序关联
# （规则定义组操作类型，比如Connect-> disconnect, File open -> File Write, visit web...）
# 每天的行为序列中，仅仅将同一个文件的File open -> File Write；Connect -> Disconnect 作为组操作类型连接
@timer
def rule_3(activity_graph, daily_sequences_list, day_delta, day_host_activity):
    rule3_num = 0
    pattern = [["File Open", "File Write"], ["Connect", "Disconnect"]]
    pattern_list = list(chain.from_iterable(pattern))

    def check_pattern(day_activity):
        # 保证是同一个obj条件下的组操作 才视为有效
        # return map{acvivity1:[[node_id1, 1], [node_id2, 1]], ...}
        res = {}
        for node_id in day_activity:
            act_type = activity_graph.nodes[node_id]['A']
            if act_type in pattern_list:
                if act_type not in res:
                    res[act_type] = []
                # 第二个值为记录是否match过节点
                res[act_type].append([node_id, 0])
        # check res
        for group_activity in pattern:
            if group_activity[0] in res and group_activity[1] in res:
                for st in res[group_activity[0]]:
                    obj = activity_graph.nodes[st[0]]['obj']

                    for ed in res[group_activity[-1]]:
                        if obj == activity_graph.nodes[ed[0]]['obj']:
                            st[1] = ed[1] = 1
                            break
        # remove all not matching nodes
        for activity in res:
            for node in res[activity]:
                if node[1] == 0:
                    res[activity].remove(node)
        
        return res

    # 将所有对应上的File Open 和 File Write都标记为1_1, 1_2, 类似2_1, 2_2

    host_day_activity = {}
    host_day_activity_pattern = {}

    for i in range(len(day_host_activity)):
        # print(day_host_activity[i])
        if day_host_activity[i] == None:
            continue
        for host in day_host_activity[i]:
            if host not in host_day_activity:
                host_day_activity[host] = []
                host_day_activity_pattern[host] = []
            host_day_activity[host].append(day_host_activity[i][host])
            host_day_activity_pattern[host].append(check_pattern(day_host_activity[i][host]))

    # 同host下，不同天之间，同组操作类型的边关联
    for host in host_day_activity_pattern:
        for day_i in range(len(host_day_activity_pattern[host])):
            for day_j in range(day_i + 1, len(host_day_activity_pattern[host])):

                for activity in host_day_activity_pattern[host][day_i]:
                    for node_i in host_day_activity_pattern[host][day_i][activity]:
                        if activity not in host_day_activity_pattern[host][day_j]:
                            continue
                        for node_j in host_day_activity_pattern[host][day_j][activity]:
                            # Todo : weight
                            rule3_num += 1
                            activity_graph.add_edge(node_i[0], node_j[0], EdgeType=3, weight=0.5)
    print("rule3 edge number : ", rule3_num)
    return activity_graph
