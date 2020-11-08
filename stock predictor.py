!pip install pandas_datareader

import pandas_datareader as pdr
key="1671e861ae3999ac70ed33886cf97c0bbd8060b8"
df=pdr.get_data_tiingo('AAPL',api_key=key)
df.to_csv("apple.csv")


df.head()

df2=df.reset_index()['close']

df2

import matplotlib.pyplot as plt
plt.plot(df2)
import numpy as np

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

df2=scaler.fit_transform(np.array(df2).reshape(-1,1))


df2.shape

training_size=int(len(df)*0.60)

test_size=len(df2)-training_size

training_size,test_size

train_data,test_data=df2[0:training_size,:],df2[training_size:len(df2),:1]

# convert to dataset matrix

def create_dataset(dataset,time_step=1):
    datax,datay=[],[]
    for i in range (len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        datax.append(a)
        datay.append(dataset[i+time_step,0])
    return np.array(datax),np.array(datay)


time_step=100
x_train,y_train=create_dataset(train_data,time_step)
x_test,y_test=create_dataset(test_data,time_step)

x_train
x_train.shape
y_train.shape

x_test.shape
y_test.shape


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer='adam')


model.summary()


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1)


###prediction with the models
train_prediction=model.predict(x_train)
test_prediction=model.predict(x_test)

train_prediction=scaler.inverse_transform(train_prediction)
test_prediction=scaler.inverse_transform(test_prediction)

import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_prediction))


math.sqrt(mean_squared_error(y_test,test_prediction))


####plotting

#shift train data
lookback=100
trainpredplot=np.empty_like(df2)
trainpredplot[:,:]=np.nan
trainpredplot[lookback:len(train_prediction)+lookback,:]=train_prediction

#shift test prediction
testpredplot=np.empty_like(df2)
testpredplot[:,:]=np.nan
testpredplot[len(train_prediction)+(lookback*2)+1:len(df2)-1]=test_prediction

plt.plot(scaler.inverse_transform(df2))
plt.plot(trainpredplot)
plt.plot(testpredplot)
plt.show()

len(test_data)

x_input=test_data[404:].reshape(1,-1)

x_input.shape

temp_input=list(x_input)

temp_input=temp_input[0].tolist()
temp_input


lstm_out=[]
n_steps=100
i=0

while(i<30):
    if (len(temp_input)<100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        # print(x_input)
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print("{} day input {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lstm_out.extend(yhat.tolist())
        i=i+1
    else:
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lstm_out.extend(yhat.tolist())
        i=i+1
print(lstm_out)



day_new=np.arange(1,101)
day_pred=np.arange(101,131)

len(df2)

df3=df2.tolist()
df3.extend(lstm_out)

plt.plot(day_new,scaler.inverse_transform(df2[1158:]))
plt.plot(day_new,scaler.inverse_transform(lstm_out))




df3=df2.tolist()
df3.extend(lstm_out)
plt.plot(df3[1200:])


df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
