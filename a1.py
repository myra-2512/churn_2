df=pd.read_csv("Churn_Modelling (1).csv")
print(df.head())
print(df.info())
print(df.describe())

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()

df['Geography']=lb.fit_transform(df['Geography'])
df['Gender']=lb.fit_transform(df['Gender'])

print(df)
print(df.info())

df=df.drop(['RowNumber','CustomerId','Surname'],axis=1)
print(df.shape)

y=df.pop('Exited')
x=df

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

print(x_train)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=10))
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))
classifier.add(Dense(units=2,kernel_initializer='glorot_uniform',activation='sigmoid'))
classifier.compile(optimizer='Adamax',loss="binary_crossentropy",metrics=['accuracy'])

model_history=classifier.fit(x_train,y_train,batch_size=10,epochs=100)

classifier.summary()

y_pred=classifier.predict(x_test)
print(y_pred())

y_pred=y_pred>0.5
print(y_pred())

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(x_test,x_train)
print(cm)

from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_train)

print(score)