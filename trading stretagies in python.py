import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime

plt.style.use("fivethirtyeight")



apple=pd.read_csv('apple.csv')


apple.head()



plt.figure(figsize=(12.5,4.5))

plt.plot(apple['adjClose'],label='apple closing price')

# plt.xticks(rotation=45)

plt.title('apple adjust close price')

plt.xlabel('09-Nov-2015 to 05-Nov-2020')

plt.ylabel('adjusted close price')

plt.legend(loc="upper left")

plt.show()



SAM30=pd.DataFrame()
SAM30["adj_cose_price"]=apple['adjClose'].rolling(window=30).mean()
SAM100=pd.DataFrame()
SAM100["adj_cose_price"]=apple['adjClose'].rolling(window=100).mean()



plt.figure(figsize=(12.5,4.5))

plt.plot(apple['adjClose'],label='apple closing price')

plt.plot(SAM30['adj_cose_price'],label='SAM30')

plt.plot(SAM100['adj_cose_price'],label='SAM100')
# plt.xticks(rotation=45)

plt.title('apple adjust close price')

plt.xlabel('09-Nov-2015 to 05-Nov-2020')

plt.ylabel('adjusted close price')

plt.legend(loc="upper left")

plt.show()



data=pd.DataFrame()
data['apple']=apple['adjClose']
data['SAM30']=SAM30['adj_cose_price']
data['SAM100']=SAM100["adj_cose_price"]

data.head()


def buy_sell(data):
    buy=[]
    sell=[]
    flag=-1
    for i in range (len(data)):
        if data['SAM30'][i] > data['SAM100'][i]:
            if flag!=1:
                buy.append(data['apple'][i])
                sell.append(np.nan)
                flag=1
            else:
                buy.append(np.nan)
                sell.append(np.nan)
        elif data['SAM30'][i] < data['SAM100'][i]:

            if flag!=0:
                buy.append(np.nan)
                sell.append(data['apple'][i])
                flag=0
            else:
                buy.append(np.nan)
                sell.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)

    return(buy,sell)


buy_sell=buy_sell(data)

buy_sell

data['buy_signal_price']=buy_sell[0]
data['sell_signal_price']=buy_sell[1]




plt.figure(figsize=(12.5,4.5))

plt.plot(data['apple'],label='apple closing price',alpha=0.35)

plt.plot(data['SAM30'],label='SAM30',alpha=0.35)

plt.plot(data['SAM100'],label='SAM100',alpha=0.35)
# plt.xticks(rotation=45)


plt.scatter(data.index,data['buy_signal_price'],label='Buy',marker='^',color='blue')

plt.scatter(data.index,data['sell_signal_price'],label='sell',marker='v',color='red')

plt.title('apple adjust close price for history buy and sell')

plt.xlabel('09-Nov-2015 to 05-Nov-2020')

plt.ylabel('adjusted close price USD')

plt.legend(loc="upper left")

plt.show()
