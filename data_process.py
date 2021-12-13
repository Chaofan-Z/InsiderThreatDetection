import os
from tqdm import tqdm

data_dir = "/mnt/188b5285-b188-4759-81ac-763ab8cbc6bf/InsiderThreatData/"
data_version = "r5.2"

# data_dir = './our_data/'
# data_version = 'r_part'

with open(os.path.join(data_dir, data_version, 'http_process.csv'), 'w') as file2:
    with open(os.path.join(data_dir, data_version, 'http.csv')) as file:
        i = 0
        for line in tqdm(file):
            i += 1
            if i % 2 :
                continue
            line = line.strip().split(',')
            line = ','.join(line[:-1])
            file2.write(line + '\n')
