import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime

plt.style.use("fivethirtyeight")

aapl=pd.read_csv('apple.csv')

appl=aapl.set_index(pd.DatetimeIndex(aapl['date'].values))

shortEMA=appl.close.ewm(span=12,adjust=False).mean()

longEMA=appl.close.ewm(span=26,adjust=False).mean()

MACD=shortEMA-longEMA

signal=MACD.ewm(span=9,adjust=False).mean()

plt.figure(figsize=(12.5,4.5))

plt.plot(appl.index,MACD,label='aapl_MACD',color='red')

plt.plot(appl.index,signal,label='signal',color='blue')

plt.xticks(rotation=45)

# plt.title('appl adjust close price')

# plt.xlabel('15-Nov-2015 to 05-Nov-2020')
#
# plt.ylabel('adjusted close price')

plt.legend(loc="upper left")

plt.show()


appl['MACD']=MACD

appl['signal']=signal

appl.head()





def buy_sell(data):
    buy=[]
    sell=[]
    flag=-1
    for i in range (len(data)):
        if appl['MACD'][i] > appl['signal'][i]:
            sell.append(np.nan)
            if flag!=1:
                buy.append(appl['close'][i])
                flag=1
            else:
                buy.append(np.nan)

        elif appl['MACD'][i] < appl['signal'][i]:
            buy.append(np.nan)
            if flag!=0:
                sell.append(appl['close'][i])
                flag=0
            else:
                sell.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)

    return(buy,sell)


buy_sell=buy_sell(appl)

buy_sell

appl['buy_signal_price']=buy_sell[0]
appl['sell_signal_price']=buy_sell[1]



plt.figure(figsize=(12.5,4.5))

plt.scatter(appl.index,appl['buy_signal_price'],label='Buy',marker='^',color='blue',alpha=1)
plt.scatter(appl.index,appl['sell_signal_price'],label='sell',marker='v',color='red',alpha=1)

plt.plot(appl['close'],label='close price',alpha=0.35)

plt.title('apple adjust close price for history buy and sell')

plt.xlabel('09-Nov-2015 to 05-Nov-2020')

plt.ylabel('adjusted close price USD')

plt.legend(loc="upper left")

plt.show()
