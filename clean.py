#File for clenaing the dataset
#Also for selecting the samples
import sys
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import random
import math

#Staring with , we shall first analyse the dataset and decide what to delete
#Firstly , we shall read the dataset
class CleanData:
    def __init__(self):
        #Now we shall read the dataset in this file
        self.old_df = pd.read_csv('zomato.csv')
        #Variable for sampling (10%)
        self.p = 0.01
    #Function for sampling data
    def data_sample(self):
        self.df = pd.read_csv('zomato.csv', header=0, skiprows=lambda i: i>0 and random.random() > self.p)
        return self.df
    #Specialised functions
    def fill_cost(self):
        self.old_df.rename(columns = {'approx_cost(for two people)':'cost'},inplace=True)
        #Now for regression to replace values
        self.old_df['cost'] = self.old_df['cost'].apply(lambda row : str(row))
        self.old_df['cost'] = self.old_df['cost'].apply(lambda row : row.replace(',',''))
        self.old_df['cost'] = self.old_df['cost'].apply(lambda row : float(row))
        rate_na_vals_df = self.old_df[self.old_df['cost'].isna()]
        rate_na_vals = rate_na_vals_df['rest_type'].to_numpy()
        #print(rate_na_vals)
        temp_df = self.old_df.dropna()
        temp_df = temp_df.loc[temp_df['rest_type'].isin(rate_na_vals)]
        X = pd.get_dummies(data=temp_df['rest_type'],drop_first=True)
        #Add the variable
        temp_df = temp_df.join(X)
        temp_df.drop(columns=['rest_type'],inplace=True)
        #Now for proper Linear regression
        Y = pd.DataFrame(temp_df['cost'])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
        regr = linear_model.LinearRegression()
        regr.fit(X_train, Y_train)
        #Only for testing purposes
        #I have used df in memory , but it is not advisable
        rest_type_x = pd.get_dummies(data = rate_na_vals_df['rest_type'],drop_first=True)
        cost_predicted = regr.predict(rest_type_x)
        cost_predicted = cost_predicted.reshape(-1)
        cost_predicted = cost_predicted.astype(float)
        #print(cost_predicted)
        cost_series = pd.Series(cost_predicted)
        #print(df.index[df.isnull().any(axis=1)])
        fill = pd.DataFrame(index=self.old_df.index[self.old_df.isnull().any(axis=1)], data=cost_predicted, columns=['cost'])
        self.old_df = self.old_df.fillna(fill)
    #Here for decoration
    def morph_rest_type(self):
        req_rest_types = ['Quick Bites' , 'Casual Dining' , 'Fine Dining' , 'Bakery' , 'Bar' , 'Food Court' , 'Pub' , 'Microbrewerry']
        #Above shall hold the unique restaraunt types we need (for double restaraunt types only)
        indi_rest_types = self.old_df['rest_type'].unique()
        new_rest_type_dict = {}
        #We shall check if it has a composite type
        for i in indi_rest_types:
            if (',' in i):
                double_type_rest = i.split(',')
                double_type_rest[1].strip(' ')
                #Checking each individual type 
                for j in double_type_rest:
                    if (j in req_rest_types):
                        new_rest_type_dict[i] = j
                        break
                #Since we do not need that many values
            #Thus the above will replace all double values with a single type
            else:
                #If some are individual , just use it as it is
                new_rest_type_dict[i] = i
        #Here we shall replace the entire thing
        self.old_df['rest_type'] = self.old_df['rest_type'].map(new_rest_type_dict)
        #For decorations
    def replace_rate(self):
        rates_list = self.old_df['rate'].to_numpy()
        #Now we shall replace the rates with their averages
        rate_count = 0
        rate_avg = 0
        #We shall replace all missing rating with the average
        #This might be biased but it works in this case
        for i in rates_list:
            if (type(i) == str and not (i == 'NEW')):
                i = i.split('/')
                #For some reason , a '-' comes out of somewhere
                if (not i[0] == '-'):
                    rate_sum_temp = float(i[0])
                    rate_avg += rate_sum_temp
            rate_count += 1
        rate_avg /= rate_count
        rate_avg = round(rate_avg,1)
        #Now we shall replace the DF with the given values
        self.old_df['rate'].fillna(str(rate_avg) + '/5', inplace=True)
        self.old_df[self.old_df['rate'] == 'NEW'].replace(str(rate_avg) + '/5')
    #For decoration
    #Function for cleaning out data
    def clean_data(self):
        self.old_df.dropna(subset=['location','rest_type','cuisines'], inplace=True)
        self.old_df.drop(['phone','dish_liked','url','listed_in(type)','reviews_list'], axis=1, inplace=True)
        #Replacing the rate
        self.replace_rate()
        self.morph_rest_type()
        self.old_df.dropna(subset=['rest_type'],inplace=True)
        self.fill_cost()
        #At the end , we shall save it to CSV again
        self.save_cleaned_file()
    #We need a function to completely clean the data
    #We need to fill the other empty rows also
    #Function for saving as CSV again
    def save_cleaned_file(self):
        self.old_df.to_csv('zomato.csv', sep=',')
#Only for testing shall I use this
#From the printing of the NA we can say
#We do not require the 'favourite-dish' hence we can remove that
#We do not require those restaraunts whose location and type are not present thus we can remove them
#There is no way to fill these values in , hence we are removing them
#Moreover two columns - phone number and dish liked are not possible to fill
#We also do not require URL and listed_in(type)
#Hence we shall remove these two columns entirely as they are not important to the analysis
df = pd.read_csv('zomato.csv')
obj = CleanData()
obj.clean_data()
#Let us consider a set of types we shall allow in the above
#The following are what we shall consider 
#Now we shall try and use regression to impute the missing values
#in the cost for two (approx)