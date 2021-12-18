from graph_contruct import get_node_from_data
from graph_embedding import get_answer
import os
import time 
from datetime import datetime
import pdb

data_dir = "/mnt/188b5285-b188-4759-81ac-763ab8cbc6bf/InsiderThreatData/"
data_version = "r5.2"
# data_version = "r_part"

sorted_vertex_list = get_node_from_data(os.path.join(data_dir, data_version), all_http=False)

# User第一次act的时间，用于计算timediff
user_start_time_dict = {}

# user上一次被采样的时间，用于采样
user_last_time_dict = {}


for node in sorted_vertex_list:
    if user_start_time_dict.get(node['sub']) is None:
        user_start_time_dict[node['sub']] = node['T']
    node['timediff'] = node['T'] - user_start_time_dict[node['sub']]

user_session_node = []
session_node = []
for node in sorted_vertex_list:
    prev_user = None

    # 理想情况下 logon开始 logoff结束    
    # if node['sub'] not in user_session:
    #     user_session[node['sub']] = []

    # make session

    if prev_user == None or prev_user == node['sub']:
        session_node.append(node)
        # session.append(node)

        if node['A'] == 'Logoff':
            user_session_node.append(session_node)
            session_node = []
    else:
        user_session_node.append(session_node)
        session_node = []

    # Todo : 
    # 1. logon开始，没有logoff结束，直接到下一个logon为止吧
    # 2. 没有logon开始，有logoff结束

label = get_answer(os.path.join(data_dir, "answers"), "r6.2" if "part" in data_version else data_version)



def check_session(session, label):
    label_number = [0] * 5
    for node_id in session:
        label_number[label.get(node_id['vertex_number'], 0)] += 1
    if label_number[0] != len(session):
        if label_number[2] != 0:
            print(label_number)
        return label_number.index(max(label_number[1:]))
    return 0



# 全量排序
user_session_node = sorted(user_session_node, key = lambda x : (x[0].__getitem__('timediff')))
# 排序后才能获取label，feat和label要对齐
user_session_label = []
user_session_label_count = {}
user_session = []
tmp = []

for session in user_session_node:
    la = check_session(session, label)
    user_session_label.append(la)
    for node in session:
        tmp.append(node['vertex_number'])
    user_session.append(tmp)
    tmp = []
    if la not in user_session_label_count:
        user_session_label_count[la] = 0
    user_session_label_count[la] += 1

print("Session number")
print(len(user_session))
print(len(user_session_label))
print(user_session_label_count)


def trans_time_to_day(time0):
    date0 = time.strptime(time0,'%m/%d/%Y %H:%M:%S')
    date0 = datetime(date0[0], date0[1], date0[2])
    return date0

def cal_day_diff(time1, time2):
    date1 = trans_time_to_day(time1)
    date2 = trans_time_to_day(time2)
    
    return (date2 - date1).days

# 进行采样
sample_user_session_node = []
min_diff = 15

print('start sample again!')
for i, session in enumerate(user_session_node):
    cur = session[0]['time']
    user = session[0]['sub']

    if user_last_time_dict.get(user) is None:
        user_last_time_dict[user] = cur 
        sample_user_session_node.append(session)
    else:
        pre = user_last_time_dict[user]
        if (trans_time_to_day(pre) == trans_time_to_day(cur)) or (cal_day_diff(pre, cur) >= min_diff) or (user_session_label[i] != 0):
            user_last_time_dict[user] = cur 
            sample_user_session_node.append(session)

                 
# 采样排序        
sample_user_session_node = sorted(sample_user_session_node, key = lambda x : (x[0].__getitem__('timediff')))

# 特别注意这里要排序后再获取label
sample_user_label = []
sample_user_session = []

sample_session_label_count = {}
tmp = []

for session in sample_user_session_node:
    la = check_session(session, label)
    sample_user_label.append(la)
    for node in session:
        tmp.append(node['vertex_number'])
    sample_user_session.append(tmp)
    tmp = []
    if la not in sample_session_label_count:
        sample_session_label_count[la] = 0
    sample_session_label_count[la] += 1


print('Sample number: ', len(sample_user_session))
print('Sample Label number: ', len(sample_user_label))
print('sample label count: \n', sample_session_label_count)







ourput_data_dir = "./output/%s/session_data/"%(data_version)
if not os.path.exists(ourput_data_dir):
    os.makedirs(ourput_data_dir)

print("Start to save sample session")
with open(ourput_data_dir + "sample_session_15", 'w') as file:
    for session in sample_user_session:
        file.write('\t'.join(session) + '\n')

print('Start to save sample label')
with open(ourput_data_dir + "sample_session_label_15", 'w') as file:
    for session_label in sample_user_label:
        file.write(str(session_label) + '\n')

# print("Start to save session and label")
# with open(ourput_data_dir + "user_session", 'w') as file:
#     for session in user_session:
#         file.write('\t'.join(session) + '\n')

# with open(ourput_data_dir + "user_session_label", 'w') as file:
#     for session_label in user_session_label:
#         file.write(str(session_label) + '\n')


