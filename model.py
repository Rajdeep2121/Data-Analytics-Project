"""
 In this file we shall create a model that predicts the cost(per two) of a restaraunt
 given the features - 'location' and 'cuisine'.
 We shall create a model using the Decision Tree classifier object that is present in the scikit learn
 module. Additionally , we shall test it and also print the accuracy.
"""
# Importing the needed libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

# Class for performing the model
class Model:
    def __init__(self):
        # Constructor for the class
        self.df = pd.read_csv('zomato.csv')

        # For dropping the unnamed column
        self.df.drop(self.df.filter(regex="Unname"),axis=1, inplace=True)

        # Getting the X and Y training data
        self.X = pd.get_dummies(self.df[['location','cuisines','rest_type']],drop_first=True)
        # self.tempX = self.df.join(self.tempX)
        self.Y = pd.DataFrame(self.df['cost']).astype(int)
    
    # Method to create the model
    def create_model(self, req_value):
        dtree = DecisionTreeClassifier()

        """
        new_value = self.df.head(1)
        print(new_value)
        new_value.loc['0','location'] = req_value['location']
        new_value.loc['0','cuisines'] = req_value['cuisines']
        new_value.loc['0','rest_type'] = req_value['rest_type']
        new_value.loc['0']
        """

        self.new_df = self.df.append(new_row,ignore_index=True)

        self.testX = pd.get_dummies(self.new_df[['location','cuisines','rest_type']],drop_first=True)

        # X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = 0.5, random_state = 0)

        # Fitting the entire model
        dtree.fit(self.X, self.Y)


        # Predicting the value
        self.dfres = dtree.predict(self.testX)
        self.score = accuracy_score(self.Y, self.dfres[:-1])
        print(self.score*100)

        self.res = self.dfres[-1]

        return self.res

# Testing the entire code
# Table columns names
# 0,address,name,online_order,book_table,rate,votes,location,rest_type,cuisines,cost,menu_item,listed_in(city)

# Excuse the poor quality of the code
new_row = {'address': '', 'name': '', 'online_order': '', 'book_table': '', 'rate': '', 'votes': '', 'location': 'Banashankari', 'rest_type': 'Fine Dining', 'cuisines': 'North Indian','cost': '', 'menu_item': '', 'listed_in': ''}
o = Model()

# check = new_df.join(mytemp)
temp = o.create_model(new_row)

print(temp)