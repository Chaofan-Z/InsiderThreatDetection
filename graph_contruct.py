import networkx as nx
import csv
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import datetime
from construct_rule import *

def get_node_from_data(dir_path, all_http=False):
    print("Get node from data : ")
    vertex_list = []
    with open(os.path.join(dir_path, "logon.csv"), 'r') as file:
        print("...logon.csv...")
    #     id,date,user,pc,activity
    #     {Q4D5-W4HH44UC-5188LWZK},01/02/2010 02:24:51,JBI1134,PC-0168,Logon
    #     {G7V0-S4TP95SA-9203AOGR},01/02/2010 02:38:28,JBI1134,PC-0168,Logoff
        read = csv.reader(file)
        next(read)
        for i in tqdm(read):
            # print(i)
            vertex_id = i[0]
            timestamp = time.mktime(time.strptime(i[1],'%m/%d/%Y %H:%M:%S'))
            
            vertex = { 'vertex_type': 'activity_logon',
                        'vertex_number': vertex_id,
                        'sub': i[2],
                        'obj': i[3],
                        'A': i[4],
                        'T': timestamp,
                        'H': i[3],
                        'time': i[1]
                        }
            vertex_list.append(vertex)

    # print(vertex_list[:5])
    with open(os.path.join(dir_path, "file.csv"), 'r') as file:
    # id,date,user,pc,filename,activity,to_removable_media,from_removable_media,content
    # {Y1W9-R7VJ77IC-9445QFNQ},01/02/2010 08:15:10,TSG0262,PC-9993,R:\79L99n6\H7RHJS5J.zip,File Open,False,True,50-4B-03-04-14 moved imaging underwent key late appearance span ontario due compiled month 07 sedins final leaders ability doug another presidents improving donation by joseph quadruple 104 agreed 16 brian upon built all to handsome searching track wounded mike march one developer owned 5000 stepping lists orange metacritic second moore supervisor currently initial
    # {Y3U8-G5BL42LO-9404XAHI},01/02/2010 08:16:01,TSG0262,PC-9993,R:\79L99n6\H7RHJS5J.zip,File Open,False,True,50-4B-03-04-14 moved imaging underwent key late appearance span ontario due compiled month 07 sedins final leaders ability doug another presidents improving donation by joseph quadruple 104 agreed 16 brian upon built all to handsome searching track wounded mike march one developer owned 5000 stepping lists orange metacritic second moore supervisor currently initial
        print("...file.csv...")
        read = csv.reader(file)
        next(read)
        for i in tqdm(read):
            # print(i)
            vertex_id = i[0]
            timestamp = time.mktime(time.strptime(i[1],'%m/%d/%Y %H:%M:%S'))
            
            vertex = { 'vertex_type': 'activity_file',
                        'vertex_number': vertex_id,
                        'sub': i[2], # user
                        'obj': i[4], # filename
                        'A': i[5], # activity
                        'T': timestamp,
                        'H': i[3], # pc,
                        'time': i[1]
                    }
            vertex_list.append(vertex)

    if all_http:
        http_file = "http.csv"
    else:
        http_file = "http_process.csv"
        
    with open(os.path.join(dir_path, http_file), 'r') as file:
    # id,date,user,pc,url,content
    # {D8Q7-C0RU46YI-7391WHNI},01/02/2010 06:46:20,HMI1448,PC-9352,http://nymag.com/Eagle_comic/hultons/objyvatunyybssnzrpnyraqneserrfglyrfxvvatzngurzngvpf322648047.jsp,eleven 1963 greater literature shorbodolio funding beating treasury both curzon single mourning huq exact visit disobeyed whose not thinking candidates necessary newly elevated eight including head those attempts present had median binds sized replacement colonial databases moderately adaptable symmetrical well drug encourage william 1840 1940s progeny possible variety 1978 on 1987 abandoned
    # {N4G0-D6NC43RD-2373QXNK},01/02/2010 06:47:25,HMI1448,PC-9352,http://nymag.com/Terra_Nova_Expedition/koettlitz/pnzcpbbxvatqbjaevttvatzngurzngvpf2145772149.asp,victims successor land restrictions provided agreeing article capture varied requests or forces 26 social medieval turkic sole population written complex visit started social down association area maulana help monument sectarian along duck jointly change words began won injured moved contract david january publish bob ready except significant appointment led making taking english true part sense entitled mothers complete fresh departure heritage youth
        print("...%s.csv..."%(http_file))
        read = csv.reader(file)
        next(read)
        for i in tqdm(read):
            vectex_id = i[0]
            timestamp = time.mktime(time.strptime(i[1],'%m/%d/%Y %H:%M:%S'))
            vertex = { 'vertex_type': 'activity_http',
                        'vertex_number': vectex_id,
                        'sub': i[2], # user
                        'obj': i[4].split(' ')[0], # url
                        'A': "visit", # activity
                        'T': timestamp,
                        'H': i[3], # pc
                        "content_list" : i[4].split(' ')[1:],
                        'time': i[1]
                    }
            vertex_list.append(vertex)

    with open(os.path.join(dir_path, "device.csv"), 'r') as file:
    # id,date,user,pc,file_tree,activity
    # {C9S1-Y8GB42VD-2923GATU},01/02/2010 07:27:19,HRE1950,PC-8025,R:\;R:\HRE1950;R:\47yHBn0;R:\54s7J45,Connect
    # {C3G4-U2ON02HC-9088IHGJ},01/02/2010 07:40:51,EMR0269,PC-6370,R:\;R:\EMR0269;R:\753Cf59;R:\18d36D6;R:\89bc6Q2,Connect
    # {X4S2-R2YC60OH-9191YYMD},01/02/2010 07:45:00,EMR0269,PC-6370,,Disconnect
        print("...device.csv...")
        read = csv.reader(file)
        next(read)
        for i in tqdm(read):
            vectex_id = i[0]
            timestamp = time.mktime(time.strptime(i[1],'%m/%d/%Y %H:%M:%S'))
            vertex = { 'vertex_type': 'activity_device',
                        'vertex_number': vectex_id,
                        'sub': i[2], # user
                        'obj': i[3], # host
                        'A': i[-1], # connect or disconnect
                        'T': timestamp,
                        'H': i[3], # pc
                        "file_tree" : i[4],
                        'time': i[1]
                    }
            vertex_list.append(vertex)

    sorted_vertex_list = sorted(vertex_list, key=lambda e: (e.__getitem__('sub'), e.__getitem__('T')))

    # print("sorted vertex list : ")
    # print(sorted_vertex_list[:1])

    return sorted_vertex_list

def get_delta_days(timestamp1, timestamp2):
    x = datetime.datetime.fromtimestamp(timestamp1) - datetime.datetime.fromtimestamp(timestamp2)
    return x.days

def get_days_from_dataset(sorted_vertex_list):
    end_time = 0
    st_time = 9999999999
    for vertex in sorted_vertex_list:
        if vertex['T'] > end_time:
            end_time = vertex['T']
        if vertex['T'] < st_time:
            st_time = vertex['T']

    print("Data delta days : ", get_delta_days(end_time, st_time)) 
    return get_delta_days(end_time, st_time) + 2

def split_node_by_day(sorted_vertex_list, day_delta, activity_graph):
    # 1000条数据大概4天

    st_time = 9999999999
    for vertex in sorted_vertex_list:
        if vertex['T'] < st_time:
            st_time = vertex['T']

    daily_sequences_list = [None] * day_delta

    print("...split node by day...")
    for vertex in tqdm(sorted_vertex_list):
        # Day of the vertex, and actual day should be increased by 1
        day_of_vertex = get_delta_days(vertex['T'], st_time) - 1

        # print(day_of_vertex)
        # If the sequence graph not exists, create it
        if not daily_sequences_list[day_of_vertex]:
            # multiGraph 无向图 可以让两个节点之间有多个边，为啥要用这个graph..
            daily_sequences_list[day_of_vertex] = nx.MultiGraph() 

        # vertex_label = 0
        # if vertex['vertex_number'] in label:
            # vertex_label = label[vertex['vertex_number']]
        daily_sequences_list[day_of_vertex].add_node(vertex['vertex_number'], type=vertex['vertex_type'],
                                                            sub=vertex['sub'], obj=vertex['obj'], A=vertex['A'],
                                                            T=vertex['T'], H=vertex['H'])
        activity_graph.add_node(vertex['vertex_number'], type=vertex['vertex_type'],
                                                            sub=vertex['sub'], obj=vertex['obj'], A=vertex['A'],
                                                            T=vertex['T'], H=vertex['H'])
    return daily_sequences_list

def construct_activity_graph():
    # 无向图，允许自循环，允许平行边
    activity_graph = nx.MultiGraph()

    # 一个用户同天同一个host时序连接
    activity_graph, host_activity = rule_1(activity_graph, daily_sequences_list, day_delta)
    # 一个用户多天同一个host的行为链的时序关联
    activity_graph = rule_2(activity_graph, daily_sequences_list, day_delta, host_activity)
    # 一个用户多天同一个host同种组操作类型时序关联
    # （规则定义组操作类型，比如Connect-> disconnect, File open -> File Write, visit web...）
    activity_graph = rule_3(activity_graph, daily_sequences_list, day_delta, host_activity)

    # test code 
    # host = 'PC-5335'
    # for day_activity in host_activity:
    #     print("Day : ", host_activity.index(day_activity))
    #     if not day_activity or host not in day_activity:
    #         continue
    #     for node_id in day_activity[host]:
    #         # print(host)
    #         print(activity_graph.nodes[node_id]['H'], activity_graph.nodes[node_id]['A'])
 
    return activity_graph

# Todo
def construct_company_graph():
    pass

# Todo
def construct_object_graph():
    pass
 
# Todo : 
# 1. construct_activity_graph
# 2. construct_company_graph
# 3. construct_object_graph
# 4. merge_graph

if __name__ == '__main__':

    st_time = time.time()

    # data_dir = "./our_data/"
    # data_version = "r_part"

    data_dir = "/mnt/188b5285-b188-4759-81ac-763ab8cbc6bf/InsiderThreatData/"
    # data_version = "r6.2"
    data_version = "r5.2"
    version = "5"

    # data_version = "r_part"
    # version = "1"

    # label = get_answer(os.path.join(data_dir, "answers"), data_version)

    sorted_vertex_list = get_node_from_data(os.path.join(data_dir, data_version))
    day_delta = get_days_from_dataset(sorted_vertex_list)

    activity_graph = nx.MultiGraph()
    daily_sequences_list = split_node_by_day(sorted_vertex_list, day_delta, activity_graph)

    # 使用一半数据构建图
    # daily_sequences_list = daily_sequences_list[:len(daily_sequences_list)]
    # activity_graph = construct_activity_graph()

    # 一个用户同天同一个host时序连接
    activity_graph, host_activity = rule_1(activity_graph, daily_sequences_list, day_delta)
    # 一个用户多天同一个host的行为链的时序关联
    activity_graph = rule_2(activity_graph, daily_sequences_list, day_delta, host_activity)
    # 一个用户多天同一个host同种组操作类型时序关联
    # （规则定义组操作类型，比如Connect-> disconnect, File open -> File Write, visit web...）
    # activity_graph = rule_3(activity_graph, daily_sequences_list, day_delta, host_activity)
    # activity_graph = rule_3_1(activity_graph, daily_sequences_list, day_delta, host_activity)

    company_graph = construct_company_graph()
    object_graph = construct_object_graph()
    print("Start save graph ")
    graph_save_path = os.path.join("./output", data_version, version, "graph")
    if not os.path.exists(graph_save_path):
        os.makedirs(graph_save_path)

    nx.write_edgelist(activity_graph, graph_save_path + "/activity_graph_edge")
    nx.write_gpickle(activity_graph, graph_save_path + "/activity_graph.gpickle")

    print("Graph save done")
    print("Time cost : ", time.time() - st_time) 
