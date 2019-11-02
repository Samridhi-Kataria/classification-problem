
# coding: utf-8

# # Problem Statement
# 

# To predict whether a bank should give person a credit card or not.
# 
# 

# # Data Description

# ID:-  A unique element for each of the customer.

# Age:- Age of the person having account in bank or the customer age.

# Experience:- Number of years they have experience in their field.

# Income:- Tells the total income of a person.

# ZIP Code:- It reperesent the zonal code for a bank. Bank in different zones have different ZIP Code.

# Family:- It represent the total number of members in their family.

# CCAvg:- It tells the cash credit average a person can take form bank.

# Education:- A numerical data which tells the education of a person in years.

# Mortgage:- A legal document signed during the loan. It tell us the number of time a person took loan from bank.

# Personal Loan:- Number of time a person took loan from bank for personal use. In this dataset it is a categorical variable. Its                 value is either 0 or 1

# Securities Account:- It represent the number of securities account a person have in a bank. It is also a categorical type of                        variable having values 0 or 1.

# CD Account:- It represent the number of CD account a person have in a bank. It is also a categorical type of                                variable having values 0 or 1.

# Online:- It tell us whether a person do some online transaction or not. Its a categorical type variable with values 0 or 1.

# CreditCard:- Its is my target variable which tell us whether a person is having a credit card or not.  

# # Importing Libraries

# In[1]:


import pandas as pd  # for reading file
from matplotlib import pyplot as plt  # for making plots
from sklearn.tree import DecisionTreeClassifier  # for using Decision tree
from sklearn.model_selection import train_test_split  # for splitting our dataset into test and train
from sklearn import metrics   # used in determining the accuracy
from sklearn.ensemble import RandomForestClassifier   #for random forest


# Reading Dataset 

# In[2]:


df=pd.read_csv('bank.csv')


# # EDA

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()   #checking null values


# In[6]:


df.describe()    #to check mean, median, min and max terms.


# As in all columns there is not a big difference between the mean and 50% (i.e median) which implies that their are very less outliars in dataset. 

# In[7]:


df.dtypes


# In[8]:


df.corr()


# Corr command tell us the realtion of different variables with each other. The value of correlation varies from [-1,1]. 

# In[9]:


df.CreditCard.value_counts()


# Tell us the number of people having credit card or not

# # Changing columns names

# In[10]:


df.columns=['ID','Age', 'Experience','Income' ,'ZIP_Code','Family','CCAvg','Education','Mortgage','Personal_Loan','Securities_Account','CD_Account','Online','CreditCard']


# Now we need to fetch our features that we can pass to our decision tree to train our model.

# # Plots

# In[11]:


plt.scatter(df.Age,df.Experience)
plt.xlabel('Age');
plt.ylabel('Experience');


# This graph shows a linear relationship between two highly corealted variables. So from both of them only one can be passed during the training of the model.

# In[12]:


df.Age.plot.density()


# From this we get that Age column in my dataset is of multi mode type and most of the ages lies between 25 to 60

# In[13]:


df[df.CreditCard==0].Age.plot.density()
df[df.CreditCard==1].Age.plot.density()


# From this graph we get that more number of people at age of 40-45(aprrox.) does not have credit card. 

# In[14]:


df[df.CreditCard==1].Income.plot.density()
df[df.CreditCard==0].Income.plot.density()


# In[15]:


plt.scatter(df.Age,df.Income)


# As from both of the graph drawn above we can get that there is no relation between age and income and very few people has income greater than 200.
# 

# In[16]:


df[df.ZIP_Code>80000].ZIP_Code.plot.density()


# From this we get the infrence that more number of people are there having ZIP Code >98000 and data is again multi modal with respect to ZIP Code

# In[17]:


df.Family.plot.density()


# Data is multi mode and more number of people are there who are single.

# In[18]:


df[df.Family==1].CreditCard.value_counts()


# In[19]:


df[df.Family==2].CreditCard.value_counts()


# In[20]:


df[df.Family==3].CreditCard.value_counts()


# In[21]:


df[df.Family==4].CreditCard.value_counts()


# By calculating the probabilities there are less chances that single person will buy a credit card on comparison with other.

# In[22]:


plt.scatter(df.Income,df.CCAvg)
plt.xlabel("Income")
plt.ylabel("CCAvg")


# There exist a linear relation between income and CCavg. People having Income less than 100 have CCAvg less than 6.

# In[23]:


df.CCAvg.plot.density()


# Data is again mullti mode w.r.t CCAvg nd maximum number of peole have CCAvg between (0,2)

# In[24]:


df.Education.value_counts()


# In[25]:


df[(df.Education==1) & (df.CreditCard==0)].CreditCard.value_counts()


# In[26]:


df[(df.Education==1) & (df.CreditCard==1)].CreditCard.value_counts()


# In[27]:


df[(df.Education==2) & (df.CreditCard==1)].CreditCard.value_counts()


# In[28]:


df[(df.Education==3) & (df.CreditCard==1)].CreditCard.value_counts()


# Different line will give us different counts of people having credit card nd their education level.

# In[29]:


df[df.Mortgage<200].Mortgage.plot.density()


# Most of the people in dataset have Mortgage=0 and few of them have some varied numbers.

# In[30]:


df[df.Mortgage==0].CreditCard.value_counts()


# People having Mortgage equal to 0 have almost 50% chance of having credit card

# In[31]:


df.Personal_Loan.value_counts()


# In[32]:


df[(df.Personal_Loan==0) & (df.CreditCard==0)].CreditCard.value_counts()


# In[33]:


df[(df.Personal_Loan==0) & (df.CreditCard==1)].CreditCard.value_counts()


# In[34]:


df[(df.Personal_Loan==1) & (df.CreditCard==0)].CreditCard.value_counts()


# In[35]:


df[(df.Personal_Loan==1) & (df.CreditCard==1)].CreditCard.value_counts()


# From these value we get that most of the person having credit card don't have personal loans. Person without any personal loan has more probability of having a credit card

# In[36]:


df.Securities_Account.value_counts()


# In[37]:


df.Securities_Account.plot.density()


# Data is bi mode either 0 or 1. It can be classified as categorical variable. Most of the people don't have Securities Account

# In[38]:


df[(df.Securities_Account==0) & (df.CreditCard==0)].CreditCard.value_counts()


# In[39]:


df[(df.Securities_Account==0) & (df.CreditCard==1)].CreditCard.value_counts()


# In[40]:


df[(df.Securities_Account==1) & (df.CreditCard==0)].CreditCard.value_counts()


# In[41]:


df[(df.Securities_Account==1) & (df.CreditCard==1)].CreditCard.value_counts()


# There is more probability that a person not having a security account will not have a credit card

# In[42]:


df.CD_Account.value_counts()


# In[43]:


df[(df.CD_Account==0) & (df.CreditCard==0)].CreditCard.value_counts()


# In[44]:


df[(df.CD_Account==0) & (df.CreditCard==1)].CreditCard.value_counts()


# In[45]:


df[(df.CD_Account==1) & (df.CreditCard==0)].CreditCard.value_counts()


# In[46]:


df[(df.CD_Account==1) & (df.CreditCard==1)].CreditCard.value_counts()


# Form these value we get that there is high probability that if person is not having a CD account he will not have a credit card
# and if he has CD account then also there is high probabilty that he will be having a Credit card.

# In[47]:


df.Online.value_counts()


# In[48]:


df[(df.Online==0) & (df.CreditCard==0)].CreditCard.value_counts()


# In[49]:


df[(df.Online==0) & (df.CreditCard==1)].CreditCard.value_counts()


# In[50]:


df[(df.Online==1) & (df.CreditCard==0)].CreditCard.value_counts()


# In[51]:


df[(df.Online==1) & (df.CreditCard==1)].CreditCard.value_counts()


# Different value counts provide us different probabilities that whether a person is having a credit card

# So here the EDA ends. We come to know different relation between different variables.

# # Separation of target variable

# In[52]:


feature_column=['Income','CD_Account','CCAvg','Age']  #these column can vary
x=df[feature_column]
y=df.CreditCard   #this is my target variable.


# # Splitting the data set into train and test
# 
# 
# 

# In[53]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# # Applying Decision Tree

# In[54]:



clf = DecisionTreeClassifier(criterion="gini", max_depth=3)

clf = clf.fit(x_train,y_train)


y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[55]:


pd.crosstab(y_test,y_pred)


# # Visualizing Tree

# In[56]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_column,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Bank_card.png')
Image(graph.create_png())


# # Applying Random Forest

# In[57]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)
model1=RandomForestClassifier(n_estimators=100,criterion='gini',random_state=50,
                               bootstrap = True,max_depth=3,
                               max_features = 'sqrt')
model1.fit(x_train,y_train)


# In[58]:



y_pred1 = model1.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))


# In[59]:


pd.crosstab(y_test,y_pred1)


# And here the code ends.
# 

# We have trained our model and its accuracy is 0.751.

# # Ends
# 

# Now as a bank I will always try to make some profit. So i will always try that a person who will be using Credit card often must get it in any case.
# Person who will not be using credit card and got it will not be a big loss for a bank but if bank is not able to give a card to person who  mihgt have used it more then its a loss.
