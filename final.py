#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sales=pd.read_csv(r'D:\11day\reg\Sales_2021.csv')


# In[7]:


sales=sales.iloc[:,:5]
sales


# In[8]:


sales.mean()


# In[ ]:





# In[ ]:





# In[9]:


sales.plot(kind='box')


# In[10]:


sales.plot(kind='hist')


# In[13]:


x=sales[['Advt','PC']]


# In[14]:


y=sales['Sales']


# In[15]:


model=ols('y~x',data=sales).fit()
model1=sm.stats.anova_lm(model)


# In[16]:


print(model1)
print(model.summary())


# In[17]:


sales['Advt'].plot(kind='box')


# In[18]:


sales['PC'].plot(kind='box')


# In[21]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

model=ols('Sales~PC',data=sales).fit()
model2=sm.stats.anova_lm(model)
print(model2)
print(model.summary())


# In[24]:


pred=model.predict()
pred


# In[56]:


res=sales['Sales'].values-pred


# In[57]:


res
res.to_numpy()


# In[58]:


# type(res)


# In[59]:


pred1=pd.DataFrame(pred,columns=['pred'])


# In[60]:


pred1


# In[61]:


res1=pd.DataFrame(res,columns=['res1'])


# In[62]:


res1


# In[63]:


from scipy import stats
zscore=stats.zscore(res)


# In[64]:


zscore


# In[71]:


zscore1=pd.DataFrame(zscore,columns=['zscore'])


# In[ ]:





# In[72]:


zscore1


# In[75]:


sales1=pd.concat([sales,res1,pred1,zscore1],axis=1)


# In[93]:


sales1.tail(50)


# In[91]:


zscore1[zscore1['zscore']>1.96]


# In[95]:


sales1.loc[sales1['zscore']>1.96,'res']=np.nan


# In[98]:


sales1.loc[sales1['zscore']<-1.96,'res']=np.nan


# In[99]:


sales.isnull().sum()


# In[100]:


sales1['res']=sales1['res'].fillna(sales1['res'].mean())


# In[102]:


sales1.drop(['zscore'],axis=1,inplace=True)


# In[104]:


sales1.drop(['res'],axis=1,inplace=True)


# In[105]:


sales


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=


# In[ ]:





# In[107]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  


# In[108]:


x_train


# In[109]:


y_train


# In[117]:


x_train1=sm.add_constant(x_train)
x_train1
model=sm.OLS(y_train,x_train1).fit()
print(model.summary())


# In[124]:


from sklearn.linear_model import LinearRegression
lg=LinearRegression()
model1=lg.fit(x_train,y_train)
y_pred=lg.predict(x_test)
y_pred
print(model1)
lg.score(x_test,y_pred)


# In[125]:


_,p_value,_,_ = sm.stats.diagnostic.het_breuschpagan(model.resid,model.model.exog)

if p_value >0.05:
    print('data is heteroscedasticity')
else:
    print('data is homoscedasticity')


# In[126]:


from sklearn import metrics


# In[129]:


mse=metrics.mean_squared_error(y_test,y_pred)
mae=metrics.mean_absolute_error(y_test,y_pred)
print('mse:',mse)
print('mae:',mae)



# In[ ]:




