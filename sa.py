import pandas as pd
import numpy as np
import math
import nltk
from alpha_vantage.timeseries import TimeSeries
from sklearn.metrics import mean_squared_error
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
from urllib.request import urlopen
from urllib.request import Request
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
from nltk.stem import SnowballStemmer
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
nltk.download('stopwords')
warnings.simplefilter('ignore')


def data(t):
    n = 1
    ticker = t
    finviz_url = 'https://finviz.com/quote.ashx?t = '
    news_tables = {}
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
    resp = urlopen(req)
    html = BeautifulSoup(resp, features="lxml")
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table
    try:
        df = news_tables[ticker]
        df_tr = df.findAll('tr')
        for i, table_row in enumerate(df_tr):
            a_text = table_row.a.text
            td_text = table_row.td.text
            td_text = td_text.strip()
            if i == n-1:
                break
    except KeyError:
        pass
    parsed_news = []
    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text()
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]

            else:
                date = date_scrape[0]
                time = date_scrape[1]

            ticker = file_name.split('_')[0]

            parsed_news.append([ticker, date, time, text])
    columns = ['Ticker', 'Date', 'Time', 'Headline']
    news = pd.DataFrame(parsed_news, columns=columns)
    return news


def review_clean(review):

    # changing to lower case
    lower = review.str.lower()

    # Replacing the repeating pattern of &#039;
    pattern_remove = lower.str.replace("&#039;", "")

    # Removing all the special Characters
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]', ' ')

    # Removing all the non ASCII characters
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+', ' ')

    # Removing the leading and trailing Whitespaces
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$', '')

    # Replacing multiple Spaces with Single Space
    multiw_remove = whitespace_remove.str.replace(r'\s+', ' ')

    # Replacing Two or more dots with one
    dataframe = multiw_remove.str.replace(r'\.{2, }', ' ')
    return dataframe


def sentiment(review):
    # Sentiment polarity of the reviews
    pol = []
    for i in review:
        analysis = TextBlob(i)
        pol.append(analysis.sentiment.polarity)
    return pol


def get_historical(quote):
    end = datetime.now()
    start = datetime(end.year-3, end.month, end.day)
    data = yf.download(quote, start=start, end=end)
    df = pd.DataFrame(data=data)
    if (df.empty):
        ts = TimeSeries(key='N6A6QT6IBFJOPJ70', output_format='pandas')
        data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
        data = data.head(503).iloc[::-1]
        data = data.reset_index()
        df = pd.DataFrame()
        df['Date'] = data['date']
        df['Open'] = data['1. open']
        df['High'] = data['2. high']
        df['Low'] = data['3. low']
        df['Close'] = data['4. close']
        df['Adj Close'] = data['5. adjusted close']
        df['Volume'] = data['6. volume']
    return df


def LSTM_ALGO(df):
    dataset_train = df.iloc[0:int(0.8*len(df)), :]
    dataset_test = df.iloc[int(0.8*len(df)):, :]
    training_set = df.iloc[:, 4:5].values
    sc = MinMaxScaler(feature_range=(0, 1))  # Scaled values btween 0,1
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []  # memory with 7 days from day i
    y_train = []  # day i
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_forecast = np.array(X_train[-1, 1:])
    X_forecast = np.append(X_forecast, y_train[-1])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # .shape 0=row,1=col
    X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, epochs=25, batch_size=32)
    real_stock_price = dataset_test.iloc[:, 4:5].values
    dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
    testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
    testing_set = testing_set.reshape(-1, 1)
    testing_set = sc.transform(testing_set)
    X_test = []
    for i in range(7, len(testing_set)):
        X_test.append(testing_set[i-7:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(real_stock_price, label='Actual Price')
    plt.plot(predicted_stock_price, label='Predicted Price')
    plt.legend(loc=4)
    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    x_input = np.array(X_test.reshape(1, -1))
    forecasted_stock_price = regressor.predict(X_forecast)
    forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
    lstm_pred = forecasted_stock_price[0, 0]
    print()
    print("##############################################################################")
    print("Tomorrow's Closing Price Prediction by LSTM: ", lstm_pred)
    print("LSTM RMSE:", error_lstm)
    print("##############################################################################")

    return lstm_pred, error_lstm


def LIN_REG_ALGO(df):
    forecast_out = int(7)
    df['Close after n days'] = df['Close'].shift(-forecast_out)
    df_new = df[['Close', 'Close after n days']]
    y = np.array(df_new.iloc[:-forecast_out, -1])
    y = np.reshape(y, (-1, 1))
    X = np.array(df_new.iloc[:-forecast_out, 0:-1])
    X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])
    X_train = X[0:int(0.8*len(df)), :]
    X_test = X[int(0.8*len(df)):, :]
    y_train = y[0:int(0.8*len(df)), :]
    y_test = y[int(0.8*len(df)):, :]
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_to_be_forecasted = sc.transform(X_to_be_forecasted)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_test_pred = y_test_pred*(1.04)
    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
    forecast_set = clf.predict(X_to_be_forecasted)
    forecast_set = forecast_set*(1.04)
    mean = forecast_set.mean()
    lr_pred = forecast_set[0, 0]
    print()
    print("##############################################################################")
    print("Tomorrow's  Closing Price Prediction by Linear Regression: ", lr_pred)
    print("Linear Regression RMSE:", error_lr)
    print("##############################################################################")
    return df, lr_pred, forecast_set, mean, error_lr


def recommending(df, global_polarity, today_stock, mean):

    if today_stock.iloc[-1]['Close'] < mean:
        if global_polarity > 0:
            idea = "RISE"
            decision = "BUY"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a", idea,
                  "in stock is expected => ", decision)
        elif global_polarity <= 0:
            idea = "FALL"
            decision = "SELL"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a", idea,
                  "in stock is expected => ", decision)
    else:
        idea = "FALL"
        decision = "SELL"
        print()
        print("##############################################################################")
        print("According to the ML Predictions and Sentiment Analysis of Tweets, a", idea,
              "in stock is expected => ", decision)
    return idea, decision


def app():
    st.title("Combining Time Series and Sentiment Analysis for  Forecasting")
    ticker = st.text_input("Enter a Stock Name")
    if st.button("Submit"):
        news = data(t=ticker)
        news.to_csv('data2.csv')
        df = pd.read_csv('data2.csv')
        df['Text'] = review_clean(df['Headline'])
        stop_words = set(stopwords.words('english'))
        df['Text'] = df['Text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
        Snow_ball = SnowballStemmer("english")
        df['Text'] = df['Text'].apply(lambda x: " ".join(Snow_ball.stem(word) for word in x.split()))
        df['sentiment_clean'] = sentiment(df['Text'])
        df.loc[(df['sentiment_clean'] > 0), 'sentiment'] = 1
        df.loc[(df['sentiment_clean'] < 0), 'sentiment'] = -1
        df.loc[(df['sentiment_clean'] == 0) | (df['sentiment_clean'] < 0.05), 'sentiment'] = 0
        df.to_csv('data.csv')
        st.write("The latest News")
        df1 = df.head(1)
        st.table(df1[["Date", "Headline"]])
        a = df1["sentiment"]
        if int(a) == 1:
            st.success("Postive News, Stock price might increases")
        elif int(a) == 0:
            st.success("Neutral News , Stock price might not change ")
        else:
            st.wrong("Negative News, Stock price might decrease")
    s = st.text_input("Enter a Stock Name for stock market prediction")
    if st.button("Predict"):
        stock = get_historical(s)
        stock.to_csv(''+s+'.csv')
        df2 = pd.read_csv(''+s+'.csv')
        today_stock = df2.iloc[-1:]
        df2 = df2.dropna()
        code_list = []
        for i in range(0, len(df2)):
            code_list.append(s)
        df3 = pd.DataFrame(code_list, columns=['Code'])
        df3 = pd.concat([df3, df2], axis=1)
        df2 = df3
        lstm_pred, error_lstm = LSTM_ALGO(df2)
        st.write("Tomorrow's Closing Price Prediction by LSTM: ", lstm_pred)
        st.write("LSTM RMSE:", error_lstm)
        df2, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df2)
        st.write("Tomorrow's Closing Price Prediction by Linear regression: ", lr_pred)
        st.write("Linear Regression RMSE:", error_lstm)
        df4 = pd.read_csv('data.csv')
        df1 = df4.head(1)
        a = df1["sentiment"]
        a = int(a)
        idea, decision = recommending(df2, a, today_stock, mean)
        st.write("According to the ML Predictions and Sentiment Analysis of News")
        st.write("Stock will ", idea)
        st.write("So you can", decision)
