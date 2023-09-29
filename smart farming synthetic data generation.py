#!/usr/bin/env python
# coding: utf-8

# #### VAE synthetic data generation 

# In[1]:


import numpy as np
#!pip install tensorflow
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd


# In[2]:


# Load the existing dataset from 'data.csv' (adjust the file path as needed)
existing_data = pd.read_csv('Crop_recommendation.csv')
existing_data.head()


# In[3]:


def Output(s):
    if s == "rice":
        return 0
    elif s== "maize":
        return 1
    elif s=="chickpea":
        return 2
    elif s=="kidneybeans":
        return 3
    if s == "mothbeans":
        return 4
    elif s== "mungbean":
        return 5
    elif s=="blackgram":
        return 6
    elif s=="lentil":
        return 7
    if s == "pomegranate":
        return 8
    elif s== "banana":
        return 9
    elif s=="mango":
        return 10
    elif s=="grapes":
        return 11
    if s == "watermelon":
        return 12
    elif s== "muskmelon":
        return 13
    elif s=="apple":
        return 14
    elif s=="orange":
        return 15
    elif s=="papaya":
        return 16
    if s == "coconut":
        return 17
    elif s== "cotton":
        return 18
    elif s=="jute":
        return 19
    elif s=="coffee":
        return 20
    elif s=="pigeonpeas":
        return 21

existing_data.label = existing_data.label.map(Output)


# In[4]:


existing_data.head()


# In[5]:


existing_data.isnull().sum()
#existing_data["MULTIPLIER"] = existing_data["MULTIPLIER"].fillna(0.0)

le_n = LabelEncoder()
existing_data['N'] = le_n.fit_transform(existing_data['N'])

le_p = LabelEncoder()
existing_data['P'] = le_p.fit_transform(existing_data['P'])

le_k = LabelEncoder()
existing_data['K'] = le_k.fit_transform(existing_data['K'])

le_temperature = LabelEncoder()
existing_data['temperature'] = le_temperature.fit_transform(existing_data['temperature'])

le_humidity = LabelEncoder()
existing_data['humidity'] = le_humidity.fit_transform(existing_data['humidity'])

le_ph = LabelEncoder()
existing_data['ph'] = le_ph.fit_transform(existing_data['ph'])

le_rainfall = LabelEncoder()
existing_data['rainfall'] = le_rainfall.fit_transform(existing_data['rainfall']) 

le_label = LabelEncoder()
existing_data['label'] = le_label.fit_transform(existing_data['label']) 

scaler = MinMaxScaler()
existing_data_scaled = scaler.fit_transform(existing_data)
X_train, X_test = train_test_split(existing_data_scaled, test_size=0.2, random_state=42)


# In[6]:


latent_dim = 8
# Encoder network
encoder_input = keras.Input(shape=(existing_data_scaled.shape[1],))
encoder_hidden = keras.layers.Dense(64, activation='relu')(encoder_input)
z_mean = keras.layers.Dense(latent_dim, name='z_mean')(encoder_hidden)
z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(encoder_hidden)


# In[7]:


def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
z = keras.layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


# Decoder network
decoder_hidden = keras.layers.Dense(64, activation='relu')(z)
decoder_output = keras.layers.Dense(existing_data_scaled.shape[1], activation='sigmoid')(decoder_hidden)


# Create the VAE model
vae = keras.Model(encoder_input, decoder_output)

# Define the custom loss function (binary cross-entropy)
def custom_bce_loss(y_true, y_pred):
    #Compute binary cross-entropy loss
    bce_loss = keras.losses.binary_crossentropy(y_true, y_pred)
    return bce_loss


# Compile the VAE model with custom loss function
vae.compile(optimizer='adam', loss=custom_bce_loss)


# Train the VAE model
epochs = 5
batch_size = 64

vae.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, X_test))
#Generate synthetic data by sampling from the VAE's latent space
num_samples = len(existing_data)
synthetic_data_encoded = np.random.normal(0, 1, size=(num_samples, latent_dim))
synthetic_data_decoded = vae.predict(synthetic_data_encoded)

#Inverse transform to obtain synthetic data in the original scale
synthetic_data = scaler.inverse_transform(synthetic_data_decoded)

#Create a DataFrame for the synthetic data
synthetic_data = pd.DataFrame(synthetic_data, columns=existing_data.columns)

#Display the first few rows of the synthetic data
print(synthetic_data.head())


#Save the synthetic data to a CSV file
#synthetic_data.to_csv('synthetic_data_vae.csv', index=False)

synthetic_data['N'] = le_n.inverse_transform(synthetic_data['N'].astype(int))
synthetic_data['P'] = le_p.inverse_transform(synthetic_data['P'].astype(int))
synthetic_data['K'] = le_k.inverse_transform(synthetic_data['K'].astype(int))
synthetic_data['temperature'] = le_temperature.inverse_transform(synthetic_data['temperature'].astype(int))
synthetic_data['humidity'] = le_humidity.inverse_transform(synthetic_data['humidity'].astype(int))
synthetic_data['ph'] = le_ph.inverse_transform(synthetic_data['ph'].astype(int))
synthetic_data['rainfall'] = le_rainfall.inverse_transform(synthetic_data['rainfall'].astype(int))
synthetic_data['label'] = le_label.inverse_transform(synthetic_data['label'].astype(int))


synthetic_data.head()


# In[8]:


synthetic_data.shape


# ### GAN Synthetic data generation 

# In[10]:


# installing the package
#!pip install ctgan


# In[11]:


gan_data = pd.read_csv('Crop_recommendation.csv')
gan_data.head()


# In[15]:


gan_data.label = gan_data.label.map(Output)


# In[16]:


from ctgan import CTGAN
columns= list(gan_data.columns)
ctgan= CTGAN(verbose= True, epochs=30)
ctgan.fit(gan_data, columns)
new_sample= ctgan.sample(10000)


# In[17]:


new_sample.shape


# In[18]:


new_sample.head()


# #### Synthetic Data through program 

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from ctgan import CTGAN
from scipy.stats import kstest
from table_evaluator import TableEvaluator
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import os
import logging
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_rows=None
pd.options.display.max_columns=None


# #### Epochs 200 

# In[2]:


data = pd.read_csv('Crop_recommendation.csv')
data.head()


# In[3]:


data = data.rename(columns={"N":"Nitogen","P":"Phosporous","K":"Potasium","ph":"Potential of Hydrogen"})
le_label = LabelEncoder()
data["label"] = le_label.fit_transform(data["label"])
data.head()


# In[6]:


def _df(data):
    df = pd.DataFrame(data)
    for c in range(df.shape[1]):
        mapping = {df.columns[c]: c}
        df = df.rename(columns=mapping)
    return df

X = (data.drop(columns=["label"])).values
y = (data["label"]).values
X = KNNImputer().fit_transform(X)
data = _df(StandardScaler().fit_transform(np.column_stack((X, y))))

data.head()



# In[10]:


tf.get_logger().setLevel(logging.ERROR)
class Gan():
    def __init__(self, data):
        self.data = data
        self.n_epochs = 200
    def _noise(self):
        noise = np.random.normal(0, 1, self.data.shape)
        return noise
    def _generator(self):
        model = tf.keras.Sequential(name="Generator_model")
        model.add(tf.keras.layers.Dense(15, activation='relu',
                    kernel_initializer='he_uniform',
                    input_dim=self.data.shape[1]))
        model.add(tf.keras.layers.Dense(30, activation='relu'))
        model.add(tf.keras.layers.Dense(
        self.data.shape[1], activation='linear'))
        return model

    def _discriminator(self):
        model = tf.keras.Sequential(name="Discriminator_model")
        model.add(tf.keras.layers.Dense(25, activation='relu',
                    kernel_initializer='he_uniform',
                    input_dim=self.data.shape[1]))
        model.add(tf.keras.layers.Dense(50, activation='relu'))
        # sigmoid => real or fake
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])

        return model

      # defining the combined generator and discriminator model,for updating the generator
    def _GAN(self, generator, discriminator):
        discriminator.trainable = False
        generator.trainable = True
        model = tf.keras.Sequential(name="GAN")
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

      # train the generator and discriminator
    def train(self, generator, discriminator, gan):
        for epoch in range(self.n_epochs):
            generated_data = generator.predict(self._noise())
            labels = np.concatenate([np.ones(self.data.shape[0]), np.zeros(self.data.shape[0])])
            X = np.concatenate([self.data, generated_data])
            discriminator.trainable = True
            d_loss , _ = discriminator.train_on_batch(X, labels)
            noise = self._noise()
            g_loss = gan.train_on_batch(noise, np.ones(self.data.shape[0]))
            print('>%d, d1=%.3f, d2=%.3f' %(epoch+1, d_loss, g_loss))
        return generator


# In[11]:


model = Gan(data=data)
generator = model._generator()
descriminator = model._discriminator()
gan_model = model._GAN(generator=generator, discriminator=descriminator)
trained_model = model.train(generator=generator, discriminator=descriminator, gan=gan_model)
noise = np.random.normal(0, 1, data.shape) 
new_data = _df(data=trained_model.predict(noise))


# In[28]:


fig, ax = plt.subplots(1, 2, figsize=(18, 6))
sns.heatmap(data.corr(), annot=True, ax=ax[0], cmap="Blues")
sns.heatmap(new_data.corr(), annot=True, ax=ax[1], cmap="Blues")
ax[0].set_title("Original Data")
ax[1].set_title("synthetic Data")


# In[13]:



fig, ax = plt.subplots(1, 2, figsize=(20, 6))
ax[0].scatter(data.iloc[:, 0], data.iloc[:, 1])
ax[1].scatter(new_data.iloc[:, 0], new_data.iloc[:, 1])
ax[0].set_title("Original Data")
ax[1].set_title("synthetic Data")


# In[15]:


table_evaluator=TableEvaluator(data, new_data)
table_evaluator.visual_evaluation()


# In[18]:



# def _df(data):
#     df = pd.DataFrame(data)
#     for c in range(df.shape[1]):
#         mapping = {df.columns[c]: c}
#         df = df.rename(columns=mapping)
#     return df


# X = (data.drop(columns=["label"])).values
# y = (data["label"]).values


# X = KNNImputer().fit_transform(X)
# data = _df(StandardScaler().fit_transform(np.column_stack((X, y))))


# # In[19]:


# tf.get_logger().setLevel(logging.ERROR)

class Gan():
    def __init__(self, data):
        self.data = data
        self.n_epochs = 20
    def _noise(self):
        noise = np.random.normal(0, 1, self.data.shape)
        return noise
    def _generator(self):
        model = tf.keras.Sequential(name="Generator_model")
        model.add(tf.keras.layers.Dense(100, activation='relu',kernel_initializer='he_uniform',input_dim=self.data.shape[1]))
        model.add(tf.keras.layers.Dense(100, activation='relu'))
        model.add(tf.keras.layers.Dense(self.data.shape[1], activation='linear'))
        return model
    def _discriminator(self):
        model = tf.keras.Sequential(name="Discriminator_model")
        model.add(tf.keras.layers.Dense(100, activation='relu',kernel_initializer='he_uniform',input_dim=self.data.shape[1]))
        model.add(tf.keras.layers.Dense(100, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        return model
    def _GAN(self, generator, discriminator):
        discriminator.trainable = False
        generator.trainable = True
        model = tf.keras.Sequential(name="GAN")
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model
    def train(self, generator, discriminator, gan):
        for epoch in range(self.n_epochs):
            generated_data = generator.predict(self._noise())
            labels = np.concatenate([np.ones(self.data.shape[0]), np.zeros(self.data.shape[0])])
            X = np.concatenate([self.data, generated_data])
            discriminator.trainable = True
            d_loss , _ = discriminator.train_on_batch(X, labels)
            noise = self._noise()
            g_loss = gan.train_on_batch(noise, np.ones(self.data.shape[0]))
            print('>%d, d1=%.3f, d2=%.3f' %(epoch+1, d_loss, g_loss))
        return generator

model = Gan(data=data)
generator = model._generator()
descriminator = model._discriminator()
gan_model = model._GAN(generator=generator, discriminator=descriminator)
trained_model = model.train(generator=generator, discriminator=descriminator, gan=gan_model)

noise = np.random.normal(0, 1, data.shape) 
new_data = _df(data=trained_model.predict(noise))


# In[19]:




fig, ax = plt.subplots(1, 2, figsize=(20, 6))
sns.heatmap(data.corr(), annot=True, ax=ax[0], cmap="Blues")
sns.heatmap(new_data.corr(), annot=True, ax=ax[1], cmap="Blues")
ax[0].set_title("Original Data")
ax[1].set_title("synthetic Data")


# In[20]:



fig, ax = plt.subplots(1, 2, figsize=(20, 6))
ax[0].scatter(data.iloc[:, 0], data.iloc[:, 1])
ax[1].scatter(new_data.iloc[:, 0], new_data.iloc[:, 1])
ax[0].set_title("Original Data")
ax[1].set_title("synthetic Data")



# In[22]:


table_evaluator=TableEvaluator(data, new_data)
table_evaluator.visual_evaluation()


# In[ ]:




data = pd.read_csv('Crop_recommendation (1).csv')
data.head()


# In[ ]:


data = data.rename(columns={"N":"Nitogen","P":"Phosporous","K":"Potasium","ph":"Potential of Hydrogen"})
le_label = LabelEncoder()
data["label"] = le_label.fit_transform(data["label"])


# In[ ]:


columns=list(data.columns)
ctgan = CTGAN(verbose=True, epochs = 5)
ctgan.fit(data,columns)


# In[ ]:


synthetic_data = ctgan.sample(10000)

