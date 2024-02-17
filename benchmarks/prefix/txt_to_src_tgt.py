import random
from os import mkdir
from os.path import join, exists

project_dir = './'
data_from = 'yelpdata_deltas_1M.txt'

lower_bound = -100.0
upper_bound = 100.0

train_split = 0.8
val_split = 0.1

with open(join(project_dir, data_from), 'r') as f:
    raw_data = f.readlines()

print(f'Number of lines: {len(raw_data)}')

filtered = []
for line in raw_data:
    src, src_speed, tgt, tgt_speed, delta = list(map(lambda x: x.strip(), line.split('\t')))

    if lower_bound <= float(delta) <= upper_bound:
        filtered.append((src, tgt, src_speed, tgt_speed, delta))

print(f'Number of lines in range: {len(filtered)}')

random.shuffle(filtered)
# split raw_data into train, valid, test
splits = {
    'train': filtered[:int(train_split*len(filtered))],
    'val': filtered[int(train_split*len(filtered)):int((train_split + val_split)*len(filtered))],
    'test': filtered[int((train_split + val_split)*len(filtered)):]
}

data_dir = join(project_dir, f'{str(lower_bound)}-{str(upper_bound)}')
if not exists(data_dir):
    mkdir(data_dir)


for data_to in ['train', 'val', 'test']:
    
    data_to_src = data_to + '.source'
    data_to_tgt = data_to + '.target'
    with open(join(data_dir, data_to_src), 'w') as f_src, \
         open(join(data_dir, data_to_tgt), 'w') as f_tgt:
        
        for src, tgt, src_speed, tgt_speed, delta in splits[data_to]:
            f_src.write(src.strip() + '\n')
            f_tgt.write(tgt.strip() + '\n')