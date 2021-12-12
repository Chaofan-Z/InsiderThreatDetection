import os

data_dir = ''
data_version = 'r6.2'

data_dir = './our_data/'
data_version = 'r_part'

with open(os.path.join(data_dir, data_version, 'http_process.csv'), 'w') as file2:
    with open(os.path.join(data_dir, data_version, 'http.csv')) as file:
        for line in file:
            line = line.strip().split(',')
            line = ','.join(line[:-1])
            file2.write(line + '\n')
