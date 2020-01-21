import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np

# addr_states: from 0~
# ['NY', 'LA', 'MI', 'WA', 'MD', 'IN', 'IL', 'FL', 'CT', 'GA', 'UT', 'NC', 'KY', 'OH', 'AR', 'OK', 'CA', 'WV', 'NJ', 'SC', 'TX', 'PA', 'KS', 'AL', 'VA', 'MO', 'AZ', 'NM', 'CO', 'RI', 'WI', 'TN', 'NV', 'MA', 'NE', 'MN', 'NH', 'OR', 'VT', 'DC', 'MS', 'ID', 'DE', 'ND', 'HI', 'ME', 'AK', 'WY', 'MT', 'SD', 'IA']
# loan_status: from 0~8
# ['Current', 'Fully Paid', 'Late (31-120 days)', 'In Grace Period', 'Charged Off', 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off']

class LoanHelper(object):
    def __init__(self, csv_file='../data/lending-club-loan-data/loan_processed.csv'):
        self.df = pd.read_csv(csv_file)
        self.df['addr_state'] = self.df['addr_state'].map(lambda x: x *1.0 / 100)
        print('read csv done')
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        loans_df = self.df.copy()
        x_feature = list(loans_df.columns)
        x_feature.remove('loan_status')
        x_val = loans_df[x_feature]
        y_val = loans_df['loan_status']
        # x_val.head() # 查看初始特征集合的数量
        y_val=y_val.astype('int')
        x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=42)
        self.data_column_name = x_train.columns.values.tolist()  # list
        self.label_column_name = x_test.columns.values.tolist()
        self.train_data = x_train.values # numpy array
        self.test_data = x_test.values
        self.train_labels = y_train.values
        self.test_labels = y_test.values

        self.dataset_train= LoanDataset(self.train_data,self.train_labels)
        self.dataset_test= LoanDataset(self.test_data,self.test_labels)

        backdoor_loan_feat = [['pub_rec_bankruptcies', 20],
                                   ['num_tl_120dpd_2m', 10],
                                   ['acc_now_delinq', 20],
                                   ['pub_rec', 100],
                                   ['tax_liens', 100],
                                   ['num_tl_90g_dpd_24m', 80]
                                   ]  # temporarily hard code
        # for loan_feat in backdoor_loan_feat:
        #     name= loan_feat[0]
        #     print(name,  x_feature.index(name))

        pos = x_feature.index('addr_state')
        dict_by_states=dict()
        for ind, data in enumerate(self.train_data):
            key=int(data[pos]*100)
            if key in dict_by_states:
                dict_by_states[key].append(ind)
            else:
                dict_by_states[key] = [ind]
        self.dict_by_states=dict_by_states
        self.state_keys=list(dict_by_states.keys())
        self.user_list= self.state_keys

        # for key in self.state_keys:
        #     print(key,len(dict_by_states[key]))

        # 16 251554
        # 21 61555
        # 25 28907
        # 13 60217
        # 9 59335
        # 0 149066
        # 20 149254
        # 26 42948
        # 14 13572
        # 5 30120
        # 10 11909
        # 6 72910
        # 24 50271
        # 12 17441
        # 33 41532
        # 11 50259
        # 28 38525
        # 29 8024
        # 48 5056
        # 23 21792
        # 34 6281
        # 15 16515
        # 7 129728
        # 3 37787
        # 22 15266
        # 18 66339
        # 35 31559
        # 37 21465
        # 19 22432
        # 31 28369
        # 32 26116
        # 40 10065
        # 42 5174
        # 30 23895
        # 2 47058
        # 8 28587
        # 4 43092
        # 17 6687
        # 1 20517
        # 27 9623
        # 49 3654
        # 44 8558
        # 41 3486
        # 38 3934
        # 39 4292
        # 45 3947
        # 36 8955
        # 47 3784
        # 46 4187
        # 43 2923
        # 50 12

        dict_by_labels = dict()
        for ind, label in enumerate(self.train_labels):
            if label in dict_by_labels:
                dict_by_labels[label].append(ind)
            else:
                dict_by_labels[label] = [ind]
        self.dict_by_labels= dict_by_labels
        self.label_keys = list(dict_by_labels.keys())

        # for key in self.label_keys:
        #     print(key,len(dict_by_labels[key]))
        # 0 735793
        # 1 833741
        # 4 209155
        # 5 2998
        # 3 7127
        # 2 17485
        # 8 613
        # 7 1596
        # 6 26


class LoanDataset(data.Dataset):
    def __init__(self, datas,labels):
        self.datas= datas
        self.labels =labels

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data, label = self.datas[index], self.labels[index]
        return data, label


if __name__ == '__main__':
    lh= LoanHelper('../data/lending-club-loan-data/loan_processed.csv')


