
# coding: utf-8

# In[2]:


import pickle
import numpy as np
from sklearn import svm
from sklearn.preprocessing import normalize


# In[ ]:


with open('results.txt', 'a') as f:
    f.write('-----------------------------------------------------------------------------\n')
    f.write('parameters default')
    f.close()


# In[3]:


rbcs = pickle.load(open('rbcs_autoencoder.pkl', 'rb'))
wbcs = pickle.load(open('wbcs_autoencoder.pkl', 'rb'))
platelets = pickle.load(open('platelets_autoencoder.pkl', 'rb'))
background = pickle.load(open('background_autoencoder.pkl', 'rb'))


# In[4]:


# l = min(len(rbcs), len(wbcs), len(platelets), len(background))


# In[5]:


#rbcs = rbcs[:l]
#wbcs = wbcs[:l]
#platelets = platelets[:l]
#background = background[:l]


# In[6]:


y_rbcs = [1 for i in range(len(rbcs))]
y_wbcs = [2 for i in range(len(wbcs))]
y_platelets = [3 for i in range(len(platelets))]
y_background = [0 for i in range(len(background))]


# In[7]:


test_x = []
test_y = []


# In[8]:


test_x = np.concatenate((rbcs[-100:], wbcs[-100:]), axis=0)
test_x = np.concatenate((test_x,platelets[-100:]), axis=0)
test_x = np.concatenate((test_x, background[-100:]), axis = 0)
# test_y = y_rbcs[-100:] + y_wbcs[-100:] + y_platelets[-100:] + y_background[-100:]


# In[9]:


test_y = y_rbcs[-100:] + y_wbcs[-100:] + y_platelets[-100:] + y_background[-100:]


# In[10]:


rbcs = rbcs[:-100]
wbcs = wbcs[:-100]
platelets = platelets[:-100]
background = background[:-100]

y_rbcs = y_rbcs[:-100]
y_wbcs = y_wbcs[:-100]
y_platelets = y_platelets[:-100]
y_background = y_background[:-100]


# In[11]:


train_x = []
train_y = []


# In[12]:


train_x = np.concatenate((rbcs, wbcs), axis=0)
train_x = np.concatenate((train_x,platelets), axis=0)
train_x = np.concatenate((train_x, background), axis = 0)


# In[13]:


# train_x = rbcs[:] + wbcs[:] + platelets[:] + background[:]
train_y = y_rbcs + y_wbcs + y_platelets + y_background


# In[14]:


train_x.shape


# In[15]:


len(train_y)


# In[17]:


train_x = np.float32(train_x)


# In[15]:


# train_x = normalize(train_x)


# In[ ]:


train_x = train_x.tolist()


# In[ ]:


clf = svm.SVC(probability=True)


# In[ ]:


clf = clf.fit(train_x, train_y)


# In[ ]:


with open('results.txt', 'a') as f:
    train_acc = clf.score(train_x, train_y)
    f.write("\ntraining accu = " + str(train_acc) + '\n')
    f.close()


# In[ ]:


test_x = np.float32(test_x)

# test_x = normalize(test_x)

test_x = test_x.tolist()

with open('results.txt', 'a') as f:
    test_acc = clf.score(test_x, test_y)
    f.write('Test accuracy = ' + str(test_acc) + '\n')
    f.write('\n')
    f.close()

s = pickle.dump(clf, open('model_autoencoder.pkl', 'wb'))
