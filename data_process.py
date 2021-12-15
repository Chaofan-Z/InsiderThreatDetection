import os
from tqdm import tqdm

# data_process 主要用于预处理http，采样没有实际作用的http，将黑名单中的http均保存下来

data_dir = "/mnt/188b5285-b188-4759-81ac-763ab8cbc6bf/InsiderThreatData/"
data_version = "r5.2"

# data_dir = './our_data/'
# data_version = 'r_part'

black_domain = ["wikileaks", 'yahoo.com', 'jobhuntersbible.com', 'boeing.com', 'linkedin.com', 'indeed.com', 'simplyhired.com', 'northropgrumman.com', 'aol.com', 'careerbuilder.com', 'raytheon.com', 'lockheedmartin.com', 'job-hunt.org', 'craigslist.org', 'hp.com', 'monster.com']

with open(os.path.join(data_dir, data_version, 'http_process.csv'), 'w') as file2:
    with open(os.path.join(data_dir, data_version, 'http.csv')) as file:
        i = 0
        for line in tqdm(file):
            line = line.strip().split(',')
            # print(line)
            url = line[4].split("/")
            # print(url)
            if len(url) < 3:
                continue

            if url[2] not in black_domain:
                i += 1
                if i % 10 :
                    continue

            line = ','.join(line[:-1])
            file2.write(line + '\n')
