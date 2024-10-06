import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
# import seaborn as sns

data = pd.read_csv('Flavor recommended system/flv.csv')

encod = LabelEncoder()
encod.fit(['Male','Female'])
data['Gender'] = encod.transform(data['Gender'])

target = 'Flavour'
x = data.drop([target],axis=1)
y= data[target]

x_train = x[:18]
x_test = x[18:]
y_train = y[:18]
y_test = y[18:]

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

def predict():
    age = int(input('Age: '))
    gender = input('Gender: ')
    if gender == "Male":
        gender = 0
    else: 
        gender = 1
    y_pred = model.predict([[age,gender]])
    print("Recommended Flavor: ", y_pred[0])

predict()