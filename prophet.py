import  pandas as pd
import fbprophet
import matplotlib.pyplot as plt


df=pd.read_csv("monthly-milk-production-pounds.csv")

df.head()

df.tail()


df.tail()

df.drop(168,axis=0,inplace=True)

df.tail()

df.plot()

df.rename(columns={"Month":"ds","Monthly milk production: pounds per cow. Jan 62 ? Dec 75":"y"},inplace=True)

df

df['ds']=pd.to_datetime(df["ds"])
df

df.head()


df["y"].plot()

df["y"]=df["y"]-df['y'].shift(1)

df
df['y'].plot()

from fbprophet import Prophet

dir(Prophet)


model=Prophet()
model.fit(df)

model

model.seasonalities


future_dates=model.make_future_dataframe(periods=365)

#prediction
prediction=model.predict(future_dates)

prediction.head()

prediction[["ds","yhat","yhat_lower","yhat_upper"]].tail()

prediction[["ds","yhat","yhat_lower","yhat_upper"]].tail()

#predction projection

model.plot(prediction)

#visualise each componenets tends and weekly

model.plot_components(prediction)

from fbprophet.diagnostics import cross_validation

df.shape

df_cv=cross_validation(model,horizon='730 days',period='180 days',initial='1095 days')
df_cv.head()


from fbprophet.diagnostics import performance_metrics
df_performance=performance_metrics(df_cv)
df_performance.head()
from fbprophet.plot import plot_cross_validation_metric
fig=plot_cross_validation_metric(df_cv,metric='mse')
