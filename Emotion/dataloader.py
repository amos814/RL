from sklearn.model_selection import train_test_split
import pandas as pd 

train_data_labeled = pd.read_csv("train_simplified.csv")
label = train_data_labeled.loc[:,"情感倾向"]
data = train_data_labeled.loc[:,"微博中文内容"]
labeled_data_list = data.values.tolist()
train_data_labeled = train_data_labeled.dropna(axis=0,how = 'any')
#数据集划分

data_train,data_test,label_train,label_test = train_test_split(data,label,test_size = 0.3,random_state = 0)

#将series的数据集类型转化为list类型
train_data = data_train.values.tolist()
test_data = data_test.values.tolist()
train_label = label_train.tolist()
test_label = label_test.tolist()

if '-' in train_label:
    train_label[train_label.index('-')] = -1
if '·' in train_label:
    train_label[train_label.index('·')] = 1
if '-' in test_label:
    test_label[test_label.index('-')] = -1
if '·' in test_label:
    test_label[test_label.index('·')] = 1

#将label中的字符串全部转化为数字
for i in range(len(train_label)):
    if isinstance(train_label[i],str):
        train_label[i] = int(train_label[i])
for i in range(len(test_label)):
    if isinstance(test_label[i],str):
        test_label[i] = int(test_label[i])