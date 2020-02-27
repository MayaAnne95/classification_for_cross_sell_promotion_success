#!/usr/bin/env python
# coding: utf-8

# In[7]:


# timeit

import timeit
code_to_test = """

# Student Name : Maya Anne
# Cohort       : Cohort 2

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports

#Import necessary packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gender_guesser.detector as gender
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer 
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz  
from IPython.display import Image 
import pydotplus 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier 



################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

original_df = pd.read_excel('Apprentice_Chef_Dataset.xlsx')




################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# use this space for all of the feature engineering that is required for your
# final model

# if your final model requires dataset standardization, do this here as well

########################
# Missing Values Treatment
########################

#Treatment of missing values and outliers 
original_df.isnull().sum()

#Creating an imputation value
fill = 'Unknown'


#Imputing 'Mas Vnr Area'
original_df['FAMILY_NAME'] = original_df['FAMILY_NAME'].fillna(fill)  

#Step 1: Splitting emails

#Placeholder list
placeholder_lst = []

#Looping over each email address
for index, col in original_df.iterrows():
    
    #Splitting email domain at '@'
    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@')
    
    #Appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

#Converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)


#Displaying the results
email_df

#Step 2: Concatenating with original DataFrame

#Safety measure in case of multiple concatenations
original_df = pd.read_excel('Apprentice_Chef_Dataset.xlsx')


#Renaming column to concatenate
email_df.columns = ['0' , 'EMAIL_DOMAIN']


#Concatenating personal_email_domain with friends DataFrame
original_df = pd.concat([original_df, email_df['EMAIL_DOMAIN']],
                     axis = 1)


#Printing value counts of personal_email_domain
original_df.loc[: ,'EMAIL_DOMAIN'].value_counts()

#Email domain types
professional_email_domains = ['@mmm.com','@amex.com','@apple.com','@boeing.com','@caterpillar.com','@chevron.com',
                              '@cisco.com','@cocacola.com','@disney.com','@dupont.com','@exxon.com','@ge.org',
                              '@goldmansacs.com','@homedepot.com','@ibm.com','@intel.com','@jnj.com','@jpmorgan.com',
                              '@mcdonalds.com','@merck.com','@microsoft.com','@nike.com','@pfizer.com','@pg.com',
                              '@travelers.com','@unitedtech.com','@unitedhealth.com','@verizon.com','@visa.com','@walmart.com']
personal_email_domains  = ['@gmail.com','@yahoo.com','@protonmail.com']
junk_email_domains  = ['@me.com','@aol.com','@hotmail.com','@live.com','@msn.com','@passport.com']

#Placeholder list
placeholder_lst = []


#Looping to group observations by domain type
for domain in original_df['EMAIL_DOMAIN']:
    
    if '@' + domain in professional_email_domains:
        placeholder_lst.append('professional')
        
    elif '@' + domain in personal_email_domains:
        placeholder_lst.append('personal')    

    elif '@' + domain in junk_email_domains:
        placeholder_lst.append('junk')


    else:
            print('Unknown')


#Concatenating with original DataFrame
original_df['DOMAIN_GROUP'] = pd.Series(placeholder_lst)


#Checking results
original_df['DOMAIN_GROUP'].value_counts()

#Creating a column that consider the seriousness of customers based on their email domain

seriousity_level_lst = []

for domains in original_df['DOMAIN_GROUP']: 
    if domains == 'professional':
        seriousity_level_lst.append(2)
        
    elif domains == 'personal':
        seriousity_level_lst.append(1) 

    elif domains == 'junk':
        seriousity_level_lst.append(0)

#Concatenating with original DataFrame
original_df['SERIOUSNESS_LEVEL'] = pd.Series(seriousity_level_lst)

#Printing our new column
original_df['SERIOUSNESS_LEVEL']

#One_hot encoding categorical variables
ONE_HOT_DOMAIN_GROUP           = pd.get_dummies(original_df['DOMAIN_GROUP'])

#Dropping categorical variables after they've been encoded
original_df = original_df.drop('EMAIL_DOMAIN', axis = 1)
original_df = original_df.drop('DOMAIN_GROUP', axis = 1)

#Joining codings together
original_df = original_df.join([ONE_HOT_DOMAIN_GROUP]) #ONE_HOT_EMAIL_DOMAIN, 

#Dropping the descrete variables 
original_df = original_df.drop('NAME', axis = 1)
original_df = original_df.drop('FAMILY_NAME', axis = 1)
original_df = original_df.drop('EMAIL', axis = 1)
original_df = original_df.drop('FIRST_NAME', axis = 1)

#Generate the Pearson correlation for all variables

original_df_corr = original_df.corr().round(2)

original_df_corr['CROSS_SELL_SUCCESS'].sort_values(ascending = False)

#Declaring explanatory variables 
original_df_better = original_df[['FOLLOWED_RECOMMENDATIONS_PCT', 'SERIOUSNESS_LEVEL', 'junk', 'MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON', 'TASTES_AND_PREFERENCES', 'professional']]

#Declaring target variables 
original_df_target = original_df.loc[ : , 'CROSS_SELL_SUCCESS']




################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# Note: Be sure to set test_size = 0.25

#Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
            original_df_better,
            original_df_target,
            test_size = 0.25,
            random_state = 2501,
            stratify = original_df_target)


# merging training data for statsmodels
original_df_train = pd.concat([X_train, y_train], axis = 1)



################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model


#Instantiating the model object without hyperparameters
rf_tuned = RandomForestClassifier(bootstrap        = False,
                                  criterion        = 'entropy',
                                  min_samples_leaf = 1,
                                  n_estimators     = 350,
                                  warm_start       = True,
                                  random_state     = 222)


#Fit step is needed as we are not using .best_estimator
rf_tuned_fit = rf_tuned.fit(X_train, y_train)


#Predicting based on the testing set
rf_tuned_pred = rf_tuned_fit.predict(X_test)



################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test) 

test_score = rf_tuned_fit.score(X_test, y_test).round(4)

"""

elapsed_time = timeit.timeit(code_to_test, number = 3)/3
print(f"""{'' * 40} Code Execution Time (seconds) : {elapsed_time}{'' * 40}""")


# In[ ]:




