
# coding: utf-8

# In[60]:

import csv
import numpy as np
import math
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


# In[2]:

def get_dict(filename, key, value):
    with open(filename) as f:
        reader = csv.DictReader(f)
        
        #create map
        dict_ = dict()
        
        for row in reader:
            key_val = row[key]
            val_val = row[value]
            dict_[int(key_val)] = int(val_val)
        
        return dict_


# In[3]:

project_map = get_dict('data/encoding/project_id_encoding.csv', 'project_id', 'mapping_number')
user_map = get_dict('data/encoding/user_id_encoding.csv', 'user_id', 'number_map')
print(len(user_map))
print(len(project_map))


# In[4]:

def read_data(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        
        #create required collections
        project_id = list()
        user_id = list()
        members = list()
        mentions = list()
        message_number = list()
        
        for row in reader:
            project_id.append(row['project_id'].rstrip().split(' '))
            user_id.append(row['user_id'].rstrip().split(' '))
            members.append(row['members'].rstrip().split(' '))
            mentions.append(row['mentions'].rstrip().split(' '))
            message_number.append(row['message_number'].rstrip().split(' '))
        if '' in members:
            print('True')
            members.remove('')
        if ' ' in mentions:
            print('True')
            mentions.remove('')
        return project_id, user_id, members, mentions, message_number


# In[5]:

def get_data_list(filename):
    project_id, user_id, members, mentions, message_number = read_data(filename)
    return project_id, user_id, members, mentions, message_number


# In[6]:

def transform_data(data_list, dict_):
    one_hot_list = np.zeros((len(data_list), len(dict_)))
    index = 0
    for sub_list in data_list:
        one_hot_encode = np.zeros((len(dict_),))
        row_number = 1
        for single_data in sub_list:
            if single_data == '0':
                continue
            if single_data != '':
                one_hot_encode[dict_[int(single_data)] - 1] = 1
        row_number += 1
        one_hot_list[index, :] = one_hot_encode
        index += 1
    return one_hot_list


# In[7]:

def count_message(message_list):
    sum_list = np.array((message_list))
    index = 0
    for sub_list in message_list:
        if '' in sub_list:
            sub_list.remove('')
        sub_list = list(map(int, sub_list))
        sum_list[index] = sum(sub_list)
        index += 1
    return sum_list


# In[8]:

def get_label(project_id, project_map):
    return transform_data(project_id, project_map)


# In[9]:

def get_user_id_data(user_id, user_map):
    return transform_data(user_id, user_map)


# In[10]:

def get_member_data(members, user_map):
    for sub_list in members:
        if '1' in sub_list:
            sub_list.remove('1')
        elif '160' in sub_list:
            sub_list.remove('160')
        elif '' in sub_list:
            sub_list.remove('')
    members_data = transform_data(members, user_map)
    return members_data


# In[11]:

def get_mention_data(mentions, user_map):
    for sub_list in mentions:
        if '1' in sub_list:
            sub_list.remove('1')
        elif '160' in sub_list:
            sub_list.remove('160')
    mentions_data = transform_data(mentions, user_map)
    return mentions_data


# In[12]:

def form_data(filename):
    project_id, user_id, members, mentions, message_number = get_data_list(filename)
    label_data = get_label(project_id, project_map)
    user_id_data = get_user_id_data(user_id, user_map)
    member_data = get_member_data(members, user_map)
    mentions_data = get_mention_data(mentions, user_map)    
    message_number = count_message(message_number)
    print('Size of user_id: ', user_id_data.shape)
    print('Size of member_data: ', member_data.shape)
    print('Size of mentions_data: ', mentions_data.shape)
    print('Size of message_number: ', message_number.shape)
    train_data = np.zeros((user_id_data.shape[0], user_id_data.shape[1] + member_data.shape[1] + mentions_data.shape[1] +  1))
    train_data[:, 0:user_id_data.shape[1]] = user_id_data
    train_data[:, user_id_data.shape[1]:user_id_data.shape[1] + member_data.shape[1]] = member_data
    train_data[:, user_id_data.shape[1] + member_data.shape[1] : train_data.shape[1] - 1] = mentions_data
    train_data[:, train_data.shape[1] - 1] = message_number
    return train_data, label_data


# In[13]:

def form_train_test_data(train_data, label_data):
    data = np.zeros((train_data.shape[0], train_data.shape[1] + label_data.shape[1]))
    data[:, 0:label_data.shape[1]] = label_data
    data[:,label_data.shape[1]:] = train_data
    np.random.shuffle(data)
    n_test = math.floor(0.1 * data.shape[0])
    print('Number of test: ', n_test)
    test = data[0:n_test,:]
    train = data[n_test:,:]
    return train, test


# In[14]:

train_data, label_data = form_data('data/Final_file.csv', )
print(train_data.shape)
print(label_data.shape)
train, test = form_train_test_data(train_data, label_data)


# Binary Relevance, Classifier Chain and Label Powerset are used to transform the multi_label problem to a single label problem.
# 
# 
# |                           | Binary Relevance | Classifier Chain | Label Powerset |
# |---------------------------|----------------- | ---------------- | -------------- |
# |  Guassian Naive Bayesian  |  0.0319          |      0.0319      |   0.1335       |
# |  Multiple Neural Network  |  0.253           |       0.255      |   0.1375       |
# |  Random Forest            |   0.263          |      0.247       |   0.269        |
# 
# 

# In[67]:

Y_train = train[:, 0:label_data.shape[1]]
X_train = train[:,label_data.shape[1]:]

Y_test = test[:, 0:label_data.shape[1]]
X_test = test[:,label_data.shape[1]:]


# ### Gaussian Naive Bayesian classification + Binary Relevance

# In[68]:

classifier = BinaryRelevance(GaussianNB())

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)

accuracy_score(Y_test, predictions)


# ### Multiple Neural Network + Binary Relevance

# In[45]:

mlp = MLPClassifier(solver='lbfgs', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=1000,verbose=10,learning_rate_init=.1)

classifier = BinaryRelevance(mlp)

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)

accuracy_score(Y_test, predictions)


# ### Gaussian Naive Bayesian + Classifier Chain

# In[26]:

classifier = ClassifierChain(GaussianNB())

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)

accuracy_score(Y_test, predictions)


# ### Neural Network + Classifier Chain

# In[44]:

mlp = MLPClassifier(solver='lbfgs', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=1000,verbose=10,learning_rate_init=.1)

classifier = ClassifierChain(mlp)

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)

accuracy_score(Y_test, predictions)


# ### Gaussian Naive Bayesian + Label Powerset

# In[29]:

classifier = LabelPowerset(GaussianNB())

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)

accuracy_score(Y_test, predictions)


# ### Neural Network + Label Powerset

# In[43]:

mlp = MLPClassifier(solver='lbfgs', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=1000,verbose=10,learning_rate_init=.1)

classifier = LabelPowerset(mlp)

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)

accuracy_score(Y_test, predictions)

