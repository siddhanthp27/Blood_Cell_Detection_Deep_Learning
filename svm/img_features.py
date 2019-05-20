
# coding: utf-8

# In[1]:


import pickle
from keras.layers import Input, Dense
from keras.models import Model
from PIL import Image
import numpy as np


# In[2]:


img_obj = Image.open('BloodImage_00002.jpg')
img_arr = np.array(img_obj)


# In[13]:


area = np.zeros(img_arr.shape[:2])


# In[15]:


area.shape


# In[16]:


features = []


# In[17]:


f = open('model_autoencoder.pkl', 'rb')
clf = pickle.load(f)
f.close()


# In[18]:


input_data = Input(shape=(19683, ))

encoding_1 = Dense(100, activation='relu')(input_data)

# encoding_2 = Dense(100, activation='relu')(encoding_1)

# decoding_1 = Dense(6400, activation='relu')(encoding_2)

decoding_2 = Dense(19683, activation='tanh')(encoding_1)

autoencoder = Model(input_data, decoding_2)


# In[19]:


autoencoder.compile(optimizer='adadelta', loss='mse')


# In[20]:


autoencoder.load_weights('autoencoder_weights.h5')


# In[21]:


new_model = Model(input=input_data, output=encoding_1)


# In[22]:


for i in range(40, 399):
    for j in range(40, 599):
        temp_features = []
        for x in range(i-40, i+41):
            for y in range(j-40, j+41):
                temp_features.append(img_arr[x][y][:])
        temp_features = np.concatenate(temp_features, axis=0)
        temp_features = np.array(temp_features).astype('float32') / 255.
        features = new_model.predict(temp_features[np.newaxis])
#         print(clf.predict(features.reshape(1, -1))[0])
        area[i][j] = clf.predict(features.reshape(1, -1))[0]
#         if (area[i][j] != 2):
        print(area[i][j])
#         print(clf.predict(features.reshape(1, -1)))
#         features.append(temp_features)


# In[23]:


f = open('00002_map.pkl', 'wb')
pickle.dump(area, f)
f.close()

