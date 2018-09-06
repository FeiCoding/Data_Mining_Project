
# coding: utf-8

# In[26]:

import csv
import tqdm


# In[27]:

def id_mapping():
    filename = 'data/metadata/user_mapping.csv'
    with open(filename) as f:
        reader = csv.DictReader(f)
        dic = dict()
        for row in reader:
            user_id = row['standup_id']
            user_name = row['slack_id']
            dic[user_name] = user_id
    return dic


# In[28]:

def channel_mapping():
    filename = 'data/metadata/channel_info.csv'
    with open(filename) as f:
        reader = csv.DictReader(f)
        dic = dict()
        for row in reader:
            channel_id = row['channel_id']
            project_id = row['project_id']
            if project_id != '':
                dic[channel_id] = project_id
    return dic        


# In[29]:

def read_data(filename):
    id_list = []
    with open(filename, encoding="ISO-8859-1") as f:
        reader = csv.DictReader(f)       
        for row in reader:
            user_id = row['user_id']
            id_list.append(user_id)
    return id_list


# In[1]:

def write_id_data(dic, id_list, filename):
    #missing_key = set()
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for id_name in tqdm.tqdm(id_list):
            if(id_name in dic):
                writer.writerow([dic[id_name]])
            else:
                writer.writerow(['XXXXX'])
                #missing_key.add(id_name)
    #return missing_key


# In[ ]:

def write_member_data(dic, id_list, filename):
    with open(filename, 'w') as f:
        writer = csv,writer(f)
        for id_name in dqdm.tqdm(id_list):


# In[31]:

def write_missing_key(missing_key, filename):    
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for key in tqdm.tqdm(missing_key):
            writer.writerow([key])


# In[3]:

def main(args):
    dic_id = id_mapping()
    dic_channel = channel_mapping()
    
    
    #------------------------------------Private Slack-----------------------------
    if args == 'Private Slack' :
        filename_private_read = 'data/Slack_Private_19062018.csv'
        filename_private_write = 'processed_data_private.csv'
        #filename_private_missing_key = 'missing_key_private.csv'
        
        id_list_private = read_data(filename_private_read)
        write_id_data(dic_id, id_list_private, filename_private_write)
        #write_missing_key(missing_key_private, filename_private_missing_key)
    
    
    
    
    #------------------------------------Public Slack-----------------------------
    if args == 'Public Slcak' :
        filename_public_read = 'data/Slack_Public_19062018.csv'
        filename_public_write = './data/processed_data_public.csv'
        #filename_public_missing_key = 'missing_key_public.csv'
        dic = id_mapping()
        id_list_public = read_data(filename_public_read)
        write_id_data(dic, id_list_public, filename_public_write)
        #write_missing_key(missing_key_public, filename_public_missing_key)
        
        
        
        
    #------------------------------------Channel Setting-----------------------------
    if args == 'Channel Setting':
        filename_channel_read = 'data/Slack_Public_19062018.csv'
        filename_channel_write = 'data/precessed_channel_data.csv'
        


# In[33]:

if __name__ == "__main__":
    args = 'Private Slack'
    main(args)


# In[ ]:



