#!/usr/bin/env python
# coding: utf-8

# # 鳶尾花(Iris)品種的辨識

# ## 載入相關套件

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ## 1. 載入資料集

# In[2]:


ds = datasets.load_iris()


# ## 2. 資料清理、資料探索與分析

# In[3]:


# 資料集說明
print(ds.DESCR)


# In[4]:


import pandas as pd
df = pd.DataFrame(ds.data, columns=ds.feature_names)
df


# In[5]:


y = ds.target
y


# In[6]:


ds.target_names


# In[7]:


# 觀察資料集彙總資訊
df.info()


# In[8]:


# 描述統計量
df.describe()


# In[9]:


# 箱型圖
import seaborn as sns
sns.boxplot(data=df)


# In[10]:


# 是否有含遺失值(Missing value)
df.isnull().sum()


# ## 繪圖

# In[11]:


# y 各類別資料筆數統計
import seaborn as sns
sns.countplot(x=y)


# In[12]:


# 以Pandas函數統計各類別資料筆數
pd.Series(y).value_counts()


# ## 3. 不須進行特徵工程

# ## 4. 資料分割

# In[13]:


# 指定X，並轉為 Numpy 陣列
X = df.values

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# 查看陣列維度
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[14]:


y_train


# ## 特徵縮放

# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# ## 5. 選擇演算法

# In[16]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()


# ## 6. 模型訓練

# In[17]:


clf.fit(X_train_std, y_train)


# ## 7. 模型評估

# In[18]:


y_pred = clf.predict(X_test_std)
y_pred


# In[19]:


# 計算準確率
print(f'{accuracy_score(y_test, y_pred)*100:.2f}%') 


# In[20]:


# 混淆矩陣
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[21]:


# 混淆矩陣圖
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)
                              , display_labels=ds.target_names)
disp.plot()
plt.show()


# ## 8. 模型評估，暫不進行

# ## 9. 模型佈署

# In[22]:


# 模型存檔
import joblib

joblib.dump(clf, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib');


# ## 10.模型預測，請參見 01_05_iris_prediction.py

# In[ ]:




