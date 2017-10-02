#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
#from sklearn.cluster import KMeans
import ScratchKMeans 
from sklearn import preprocessing, cross_validation
import pandas as pd

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_csv('titanic.csv')
print(df.head())

df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())
def handle_non_numerical_data(df):
    #colums df
    columns = df.columns.values
    for column in columns:
        #each column create a list of unique values
        text_digit_vals = {}
        def convert_to_int(val):
           return text_digit_vals[val]
#       looking for data type within the olumn which is not a number value
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
           # print(df[column].name)
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            #unique values given number id as new value to replace text with a int
            for unique in unique_elements:
               # print(unique)
                #check if value exists in list of unique values
                 #eg [bob,1],[glob,2],[frog,3]
                if unique not in text_digit_vals:
                    #increment new unique value within the list
                   
                    text_digit_vals[unique] = x
                    x+=1
#           Call convert to int inside columns 
            df[column] = list(map(convert_to_int, df[column]))

    return df
df = handle_non_numerical_data(df)
print(df.head())
#X is data - survived
X = np.array(df.drop(['survived'], 1).astype(float))
#convert to scala for improved accuracy
X = preprocessing.scale(X)
#y is the value of survived to be checked against after the classifier makes a prediction eg test data for the model
y = np.array(df['survived'])
#Call MATH MAGIC random nodes selected then eucidian distance across whole data set BOOM and it just works... mental
#---------------------------------Insert My K_Means--------------------------
clf = ScratchKMeans.K_Means()
#pickle time?
#------------------Use My fitness --------------------------
clf.fit(X)
#self learning with n_clusters=2 [1,0] randomly assigned
correct = 0
for i in range(len(X)):
    #formatting
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    #looking in the dataset for the classification created by clf = KMeans(n_clusters=2)
    #clf.fit(X)
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1

print(correct/len(X))
