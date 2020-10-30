# Importing liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Datasets
dataset = pd.read_csv('Python/Data.csv')

# Seperating dependent and independent vairiable
# Independent vairiable
x = dataset.iloc[:,:-1].values

#Dependent vairible
y = dataset.iloc[:,3].values




# Dealing with the missing data by replacing the missing data by the mean of that corresponding column

# we will use SimpleImputer from scikit-learn liberary.
from sklearn.impute import SimpleImputer 
# Instanciating the imposter class
imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean')
# selecting the columns with the missing data
imputer.fit(x[:,1:3])
# Replacing the missind data withe the mean of the corresponding column's mean
x[:,1:3] = imputer.transform(x[:,1:3])




# Encoding Categorical Data

# we will use LabelEncoder from scikit-learn liberary.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# encoding categorical data in array x
labelencoder_x = LabelEncoder() 
# specifing the index of the columns having categorical vairiables in array x 
x[:,0] = labelencoder_x.fit_transform(x[:,0]) 

# now emplimenting Dummy Encoding so that ML algorithm model do not attribute an order in to the categorical vairiable so that we can have the correct computation while modeling dataset 
# The below code is deprecated now so we need to use the ColumnTransformer "------ from sklearn.compose import ColumnTransformer -------"
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray() 

# encoding categorical data in array y
labelencoder_y = LabelEncoder() 
# specifing the index of the columns having categorical vairiables in array y 
y = labelencoder_y.fit_transform(y) 




# Splitting the dataset into Training set and Test set

# Earlier we used to use cross_validator but now it is deprecated
# from sklearn.cross_validation import train_test_split

# so now we use model_selection
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)











