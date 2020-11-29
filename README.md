# Data-Analytics-Project

File to clean the dataset has been added.
The dataset is too large to be shared via GitHub hence you need to download it on your own 
from the website link here - https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants

Additionally we shall use a small sample of the dataset for visualisation (around 10%).


CLEANING:
I have more or less cleaned the dataset.
Only thing is approx_cost needs to be cleaned , but that is simple.
We might need to use a model to clean it.
Code is stored in 'clean.py'

MODEL:
Added an sklearn model that uses a Decision Tree to estimate the cost of a restaraunt given 
features such as location , cuisine and type of restaraunt.
Additionally a regression model has been used to clean 'ratings' column.

VISUALISATIONS:
A wide variety of visualisations have been performed on the dataset , all are present in the Jupyter Notebook file.

STEPS ON RUNNING THE CODE:
Ensure that the CSV is in the same folder or directory as all the python codes.
If dataset is unclean , do the following -
python3 clean.py
The above will clean the dataset and save it as it is under the same name.
Once done , run the Jupyter Notebook to obtain the different visualisations.

To run the model simply type -
python3 model.py


NOTE - Dataset must be names as 'zomato.csv' else the code will not run.
