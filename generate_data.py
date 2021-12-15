from graph_contruct import get_node_from_data
from graph_embedding import get_answer
import os

data_dir = "/mnt/188b5285-b188-4759-81ac-763ab8cbc6bf/InsiderThreatData/"
data_version = "r5.2"
# data_version = "r_part"

sorted_vertex_list = get_node_from_data(os.path.join(data_dir, data_version), all_http=False)

user_session = []
session = []
for node in sorted_vertex_list:
    prev_user = None

    # 理想情况下 logon开始 logoff结束    
    # if node['sub'] not in user_session:
    #     user_session[node['sub']] = []

    # make session

    if prev_user == None or prev_user == node['sub']:
        session.append(node['vertex_number'])
        # session.append(node)

        if node['A'] == 'Logoff':
            user_session.append(session)
            session = []
    else:
        user_session.append(session)
        session = []
    
    # Todo : 
    # 1. logon开始，没有logoff结束，直接到下一个logon为止吧
    # 2. 没有logon开始，有logoff结束

label = get_answer(os.path.join(data_dir, "answers"), "r6.2" if "part" in data_version else data_version)

user_session_label = []
user_session_label_count = {}

def check_session(session, label):
    label_number = [0] * 5
    for node_id in session:
        label_number[label.get(node_id, 0)] += 1
    if label_number[0] != len(session):
        if label_number[2] != 0:
            print(label_number)
        return label_number.index(max(label_number[1:]))
    return 0

for session in user_session:
    la = check_session(session, label)
    user_session_label.append(la)
    if la not in user_session_label_count:
        user_session_label_count[la] = 0
    user_session_label_count[la] += 1

print("Session number")
print(len(user_session))
print(len(user_session_label))
print(user_session_label_count)

ourput_data_dir = "./output/%s/session_data/"%(data_version)
if not os.path.exists(ourput_data_dir):
    os.makedirs(ourput_data_dir)

print("Start to save session and label")
with open(ourput_data_dir + "user_session", 'w') as file:
    for session in user_session:
        file.write('\t'.join(session) + '\n')

with open(ourput_data_dir + "user_session_label", 'w') as file:
    for session_label in user_session_label:
        file.write(str(session_label) + '\n')


