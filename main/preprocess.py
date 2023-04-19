import numpy as np
from config import HP
import os

train_set_ratio = 0.7
dev_set_ratio = 0.2
test_set_ratio = 0.1

np.random.seed(HP.seed)
data_set = np.loadtxt(HP.data_path, delimiter=',')
np.random.shuffle(data_set)

n_items = data_set.shape[0]

train_set_num = int(train_set_ratio * n_items)
dev_set_num = int(dev_set_ratio * n_items)
test_set_num = n_items - train_set_num - dev_set_num

np.savetxt(os.path.join(HP.data_dir, 'train.txt'), data_set[:train_set_num], delimiter=',')
np.savetxt(os.path.join(HP.data_dir, 'dev.txt'), data_set[train_set_num:train_set_num + dev_set_num], delimiter=',')
np.savetxt(os.path.join(HP.data_dir, 'test.txt'), data_set[train_set_num + dev_set_num:], delimiter=',')
