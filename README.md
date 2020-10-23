# Data-Analytics-Project

File to clean the dataset has been added.
The dataset is too large to be shared via GitHub hence you need to download it on your own 
from the website link here - https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants

Additionally we shall use a small sample of the dataset for visualisation (around 10%).

The three files I have created are for their purposes as shown by the name.

CLEANING:
I have more or less cleaned the dataset.
Only thing is approx_cost needs to be cleaned , but that is simple.
We might need to use a model to clean it.
Code is stored in 'clean.py'

MODEL:
Added a basic sklearn model that uses a Decision Tree to estimate the cost of a restaraunt given 
features such as location , cuisine and type of restaraunt.
However it might not be perfect , if possible please add more models for prediction of features such as - rate , location (if possible)