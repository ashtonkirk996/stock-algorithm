import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

#select the stock to predict
stock = yf.Ticker("AMZN")

#get the historical stock data 
stock = stock.history(period="max")

#clean the data
del stock["Dividends"]
del stock["Stock Splits"]
stock = stock.loc["2000-01-01":].copy()

#setup target (will the price go up or down tomorrow?)
stock["Tomorrow"] = stock["Close"].shift(-1)
stock["Target"] = (stock["Tomorrow"] > stock["Close"]).astype(int)

# train machine learning model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

#put all but the last 100 rows into the training data, then use the last 100 rows to test
#train = stock.iloc[:-100]
#test = stock.iloc[-100:]

predictors =["Close","Volume", "Open", "High", "Low"]


def predict(train, test, predictors, model:RandomForestClassifier):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data:pd.DataFrame, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
        
    return pd.concat(all_predictions)



#calculate the mean trading price over each time period
horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = stock.rolling(horizon).mean()

    ratio_column =f"Close_Ratio_{horizon}"
    stock[ratio_column] = stock["Close"]

    trend_column = f"Trend_{horizon}"
    stock[trend_column] = stock.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

stock = stock.dropna()

predictions = backtest(stock, model, predictors)
print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))
print(predictions["Target"].value_counts()/predictions.shape[0])