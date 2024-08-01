import os

import sklearn
import pandas as pd
from sklearn.model_selection import KFold

path_csv = pd.read_csv('//mnt/orton/PD-L1/EBER_TCGA/TCGA_label.csv')

names = path_csv['slide_id'].tolist()
kf = KFold(n_splits=5,shuffle=True,random_state=2)
result_dir = '/mnt/orton/codes/CLAM_zoo/CLAM/splits/TCGA_my_sed2/'
os.makedirs(result_dir,exist_ok=True)
m = 0
for train_index, test_index in kf.split(names):
    X_train=[names[i] for i in train_index]
    X_test=[names[i] for i in test_index]
    max_length = len(X_train)
    X_test.extend([None] * (max_length - len(X_test)))
    data = {'train': X_train, 'val': X_test,'test': X_test}
    pdds = pd.DataFrame(data)
    # pdds['train']= train
    # # pdds['val'] =  X_test
    # # pdds['test'] = X_test

    # pdds.loc[X_test, 'val'] =len(X_test)
    # pdds.loc[X_test, 'test'] = len(X_test)

    pdds.to_csv(result_dir + 'splits_'+str(m)+'.csv')
    m +=1
print('fds')