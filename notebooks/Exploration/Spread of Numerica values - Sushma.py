#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import matplotlib.pyplot as plot
import seaborn as se

df = pd.read_csv("C:\\Users\\kiran\\OneDrive\\Desktop\\StepWeek5\\AISCB_FraudDetection\\data\\interim\\fraud_detection_null_cleaned.csv")

print(df.info())


# In[17]:


columns = ['DAYS_ID_PUBLISH',           
   'FLAG_MOBIL',                  
   'FLAG_EMP_PHONE',         
   'FLAG_WORK_PHONE',              
   'FLAG_CONT_MOBILE',           
   'FLAG_PHONE',                   
   'FLAG_EMAIL',                   
   'CNT_FAM_MEMBERS',            
   'REGION_RATING_CLIENT',        
  'REGION_RATING_CLIENT_W_CITY' ]


for i, col in enumerate(columns):
     plot.subplot(5,2,i+1)
     # plot.scatter(x=df[col], y=df['TARGET'])
     se.histplot(df[col])
plot.subplots_adjust( left= 0.065,
    bottom=0.072,
    right=0.952,
    top=0.935,
    wspace=0.143,
    hspace=0.564)
plot.show()


# In[18]:


outlier_colors = {'markerfacecolor': 'r', 'marker': 'd'}

for i,col in enumerate(columns):
     plot.subplot(5,2,i+1)
     plot.boxplot(df[col],vert = False, flierprops=outlier_colors)
     plot.title(col)
plot.subplots_adjust( left= 0.065,
    bottom=0.072,
    right=0.952,
    top=0.935,
    wspace=0.143,
    hspace=0.564)
plot.show()


# In[19]:


for i,col in enumerate(columns):
    plot.subplot(5,2,i+1)
    se.scatterplot(x=df[col],y=df['TARGET'])
    plot.xlabel(col)
    plot.ylabel('TARGET')
plot.subplots_adjust( left= 0.065,
    bottom=0.072,
    right=0.952,
    top=0.935,
    wspace=0.143,
    hspace=0.564)
plot.show()


# In[28]:


aggre = df.groupby(['CODE_GENDER']).mean(numeric_only = True)
print(aggre)

colors = ['#FFB5E8', '#6EB5FF','#BFFCC6']

for i,col in enumerate(columns):
    plot.subplot(5,2,i+1)
    aggre[col].plot.bar(color = colors)
plot.subplots_adjust( left= 0.065,
    bottom=0.072,
    right=0.952,
    top=0.935,
    wspace=0.143,
    hspace=0.564)
plot.show()


# In[26]:


for i, col in enumerate(columns):
    plot.subplot(5, 2, i+1)
    # Separate the data into two groups: 0s and 1s
    se.distplot(df[df[col] == 0][col], kde=False, color='#6EB5FF', bins=[-0.5, 0.5, 1.5])
    se.distplot(df[df[col] == 1][col], kde=False, color='#FFB5E8', bins=[-0.5, 0.5, 1.5])
    plot.title(col)
    plot.xticks([0, 1])  # Ensuring that we only have ticks at 0 and 1

plot.subplots_adjust(left=0.065, bottom=0.072, right=0.952, top=0.935, wspace=0.143, hspace=0.564)
plot.show()


# In[ ]:




