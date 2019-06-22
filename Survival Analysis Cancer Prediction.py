
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_b2ad6ec7673044d0b104242b67ffa289 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='U8oWrS56jzbSzrmgyHejwt6eD_5RZMd7l4T94ugDKLCk',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_b2ad6ec7673044d0b104242b67ffa289.get_object(Bucket='survivalanalysiscancerprediction-donotdelete-pr-h1bqvo24bgyej9',Key='haberman.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()



# In[3]:


dataset


# In[4]:


dataset.corr()


# In[5]:


sns.heatmap(dataset.corr(),annot=True)


# In[6]:


x=dataset.iloc[:,:3].values


# In[7]:


x


# In[8]:


x.ndim


# In[9]:


y=dataset.iloc[:,3].values


# In[10]:


y


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8,random_state=0)


# In[13]:


x_train


# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


lr=LogisticRegression()


# In[16]:


lr.fit(x_train,y_train)


# In[17]:


y_pred=lr.predict(x_test)


# In[18]:


y_pred


# In[19]:


from sklearn.metrics import accuracy_score


# In[20]:


accuracy_score(y_test,y_pred)


# In[21]:


y_pred=lr.predict([[30,57,6]])


# In[22]:


y_pred


# In[23]:


y_pred=lr.predict([[30,57,3]])


# In[24]:


y_pred


# In[25]:


plt.scatter(x_train[:,0],y_train)


# In[26]:



get_ipython().system(u'pip install watson-machine-learning-client --upgrade #to deploy')


# In[27]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[28]:


wml_credentials={"instance_id": "41023c87-4e9b-4082-ab2b-3b7567a15409",
  "password": "923c24d6-6bd7-4c78-9a3e-c7749e7b4b91",
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "username": "6b4becde-de25-4974-9a72-3a672638dee0","accesskey": "kG8y8fajMnIVYPW2UAIX4t9-msn9Bwyh-qrDkwVNbwVm"
}


# In[29]:


client=WatsonMachineLearningAPIClient(wml_credentials)


# In[30]:


import json #json=javascript object notation
instance_details=client.service_instance.get_details()
print(json.dumps(instance_details,indent=2))


# In[31]:


model_props={client.repository.ModelMetaNames.AUTHOR_NAME:"Manasa",
            client.repository.ModelMetaNames.AUTHOR_EMAIL:"maggipinky1999@gmail.com",
            client.repository.ModelMetaNames.NAME:"Survival Analysis Cancer Prediction"} 


# In[32]:



model_artifact=client.repository.store_model(lr,meta_props=model_props)


# In[33]:



published_model_uid=client.repository.get_model_uid(model_artifact)


# In[34]:


published_model_uid


# In[ ]:


created_deployment=client.deployments.create(published_model_uid,name="Survival Analysis Cancer Prediction")


# In[ ]:



scoring_endpoint=client.deployments.get_scoring_url(created_deployment)
scoring_endpoint

