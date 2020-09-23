#File for clenaing the dataset
#Also for selecting the samples
import sys
import os
import pandas as pd
import numpy as np
import sklearn
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
        #self.old_df.dropna(subset=['location','rest_type','cuisines'], inplace=True)
        #self.old_df.drop(['phone','dish_liked','url','listed_in(type)'], axis=1, inplace=True)
        #Replacing the rate
        self.replace_rate()
        #At the end , we shall save it to CSV again
        self.save_cleaned_file()
    #Function for saving as CSV again
    def save_cleaned_file(self):
        self.old_df.to_csv('zomato_cleaned.csv', sep=',')
#Only for testing shall I use this
#From the printing of the NA we can say
#We do not require the 'favourite-dish' hence we can remove that
#We do not require those restaraunts whose location and type are not present thus we can remove them
#There is no way to fill these values in , hence we are removing them
#Moreover two columns - phone number and dish liked are not possible to fill
#We also do not require URL and listed_in(type)
#Hence we shall remove these two columns entirely as they are not important to the analysis
df = pd.read_csv('zomato_cleaned.csv')
print(df.isna().sum())

