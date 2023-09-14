import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../../data/playNoPlay.csv")
x= df[['outlook','temp','humidity','windy']] #
y = df['play']
le= LabelEncoder()
y= le.fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0)

encoder = ce.OrdinalEncoder(cols=['outlook','temp','humidity','windy'])
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)
nsvc = SVR()
nsvc.fit(x_train, y_train)

score = nsvc.score(x_train, y_train)
print("Score: ", score)