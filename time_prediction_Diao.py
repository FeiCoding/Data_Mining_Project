
# coding: utf-8

# In[15]:

import numpy as np
import csv
import matplotlib.pyplot as plt
import tqdm
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


# In[16]:

poly = PolynomialFeatures(2)
scaler = StandardScaler()


# In[9]:

def form_data():
    filename = 'data/all_standups_jun17_jun18.csv'
    user_id = np.zeros((15005, 1))
    interest_id = np.zeros((15005, 1))
    project_id = np.zeros((15005, 1))
    time = np.zeros((15005, 1))
    index = 0;
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id[index, :] = row['user_id']
            interest_id[index, :] = row['interest_id']
            project_id[index, :] = row['project_id']
            time[index, :] = row['time']
            index += 1
    
    n = time.shape[0]
    data = np.zeros((n, 3))    
    data[:, 0] = user_id[:,0]
    data[:, 1] = interest_id[:,0]
    data[:, 2] = project_id[:,0]
    
    data = poly_features(data)
    data = normalization(data)
    print(data.shape)
    data = np.append(data, time, axis=1)
    np.random.shuffle(data)
    print(data.shape)
    return data


# In[10]:

def normalization(data):
    scaler.fit(data)
    trans_data = scaler.transform(data)
    return trans_data


# In[11]:

def poly_features(data):
    dataTransform = poly.fit_transform(data)
    return dataTransform


# In[ ]:

def linear_regression_model(train, test):
    model = LinearRegression(normalize = True)
    label_length = train.shape[1] - 1
    x = train[:, 0:label_length]
    y = train[:, label_length]  
    model.fit(x, y)
    #for i in range(test.shape[0]):
        #print(test[i, label_length].reshape(-1, 1) - model.predict(test[i, 0:label_length]).reshape(-1, 1))
    error = np.sum(mean_squared_error(test[:, label_length], model.predict(test[:, 0:label_length]))) / test.shape[0]
    return error, model


# In[34]:

def radom_forest_regressor(train, test):
    model = RandomForestRegressor()
    label_length = train.shape[1] - 1
    x = train[:, 0:label_length]
    y = train[:, label_length]  
    model.fit(x, y)
    #for i in range(test.shape[0]):
        #print(test[i, label_length].reshape(-1, 1) - model.predict(test[i, 0:label_length]).reshape(-1, 1))
    error = np.sum(mean_squared_error(test[:, label_length], model.predict(test[:, 0:label_length]))) / test.shape[0]
    return error, model


# In[35]:

def SVM_regressor(train, test):
    model = SVR()
    label_length = train.shape[1] - 1
    x = train[:, 0:label_length]
    y = train[:, label_length]  
    model.fit(x, y)
    #for i in range(test.shape[0]):
        #print(test[i, label_length].reshape(-1, 1) - model.predict(test[i, 0:label_length]).reshape(-1, 1))
    error = np.sum(mean_squared_error(test[:, label_length], model.predict(test[:, 0:label_length]))) / test.shape[0]
    return error, model


# In[36]:

def form_test_train(data, n):
    test_size = math.floor(n * 0.1)
    test = np.zeros((test_size, data.shape[1]))
    train = np.zeros((n - test_size, data.shape[1]))
    test = data[0:test_size, :]
    train = data[test_size:, :]
    return test, train


# In[37]:

def data_pre_process():
    data = form_data()    
    test, train = form_test_train(data, data.shape[0])
    return train, test


# In[44]:

def single_test(data, model):
    print('test normalized data: ' ,data.shape)
    return model.predict(data)


# In[47]:

def main():
    train, test = data_pre_process()
    
    Poly_error, polynomial_model = radom_forest_regressor(train, test)    
    print('Random Forest Average Error: ', Foreast_error)
    Foreast_error, random_forest_model = radom_forest_regressor(train, test)    
    print('Random Forest Average Error: ', Foreast_error)
    SVR_errors, SVR_model = SVM_regressor(train, test)
    print('SVM Regression Average Error: ', SVR_errors)
    
    x = poly.transform(np.array([3, 29, 2]).reshape(1, -1).astype(float))
    x = scaler.transform(x)
    #print(x)
    #print(test[0,:])
    print(single_test(x, random_forest_model))


# In[48]:

if __name__ == "__main__":
    main()


# In[ ]:



