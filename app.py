import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

# Scaler
scaler = MinMaxScaler(feature_range = (0, 1))

# Loading the trained model
model = load_model('keras_model.h5')

start = '2014-01-01'
end = '2024-01-31'

st.title('Stock Price Prediction')


user_input = st.text_input('Enter the Stock Ticker', 'TATAMOTORS.NS')
# df = yf.download(user_input, start , end)

# Predict one value
# <---------------------------------------------------------------->

sDate = st.date_input("Enter start date")
eDate = st.date_input("Enter end date")

df = yf.download(user_input, sDate, eDate)

quote_df = df.filter(['Close'])
last_100_days = quote_df[-100:].values
last_100_days_scaled = scaler.fit_transform(last_100_days)
X_test = []
X_test.append(last_100_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price.tolist()
# pred_price = str(pred_price[0][0])

st.session_state.disabled = True

css = """
    <style>
    .custom-input {
        background-color: #262730;
        border: none;
        color: #FAFAFA;
        padding: 7px 10px;
        text-align: left;
        font-size: 16px;
        margin: 4px 2px;
        cursor: not-allowed;
        opacity: 1; /* Adjust opacity to control the text visibility */
        width: 100%;
        border-radius: 8px;
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

st.subheader('Predicted Price')
st.markdown(f'<input type="text" class="custom-input" value="{pred_price[0][0]}" disabled>', unsafe_allow_html=True)

# <---------------------------------------------------------------->

# Describing the data
st.subheader('Data from 2014 - 2024')
st.dataframe(df, width=710)
# st.write(df.describe())

# Visualizations
st.subheader('Closing Price VS Time Chart')
close_price_time_chart = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
st.pyplot(close_price_time_chart)

df = df.reset_index()
date_values = df['Date'].to_numpy()
date_values = date_values[:10]
close_value = df['Close'].to_numpy()
close_values = close_value[:10]

st.subheader('Bar Graph for visualizing Closing Price')
bar_graph = plt.figure(figsize = (12, 6))
plt.bar(date_values, close_values, width = 0.5)
plt.xlabel('Dates')
plt.ylabel('Closing Price')
plt.title('Prices with date')
st.pyplot(bar_graph)

st.subheader('Closing Price VS Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
values = {'Closing Price' : df.Close, '100MA' : ma100}
values = pd.DataFrame.from_dict(values)
st.dataframe(values, width=710)

ma100 = df.Close.rolling(100).mean()
mov_avg_100 = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r', label = 'Moving Average of 100 days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(mov_avg_100)

st.subheader('Closing Price VS Time Chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
values = {'Closing Price' : df.Close, '100MA' : ma100, '200MA' : ma200}
values = pd.DataFrame.from_dict(values)
st.dataframe(values, width=710)

ma200 = df.Close.rolling(200).mean()
mov_avg_200 = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r', label = 'Moving Average of 100 days')
plt.plot(ma200, 'g', label = 'Moving Average of 200 days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(mov_avg_200)

# Splitting the data into training and testing datasets
data_training = pd.DataFrame(df['Close'][0 : int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

# scaler = MinMaxScaler(feature_range = (0, 1))

data_training_array = scaler.fit_transform(data_training)

# ----------------------------------------------------------------

# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

# scaler = scaler.scale_

# scaler_factor = 1/scaler[0]
# y_predicted = y_predicted * scaler_factor
# y_test = y_test * scaler_factor

# y_predicted = y_predicted.flatten()

y_test = y_test.reshape(-1, 1)

y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test)

y_predicted = y_predicted.flatten()
y_test = y_test.flatten()

# Final Graph
st.subheader('Predicted VS Original')
values = {"Original" : y_test, "Predicted" : y_predicted}
values = pd.DataFrame.from_dict(values)
st.dataframe(values, width=710)

pred_vs_org = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(pred_vs_org)

# Grouped Bar Graph
y_predicted = y_predicted[:5]
y_test = y_test[:5]

categories = ['Price 1', 'Price 2', 'Price 3', 'Price 4', 'Price 5']
bar_width = 0.35
  
        # --> Set the positions of the bars on the x-axis
r1 = np.arange(len(categories))
r2 = [x + bar_width for x in r1]

grouped_bar_graph = plt.figure(figsize = (12, 6))

        # --> Create the grouped bar plot
plt.bar(r1, y_test, color='blue', width=bar_width, edgecolor='grey', label='Original')
plt.bar(r2, y_predicted, color='red', width=bar_width, edgecolor='grey', label='Predicted')
  
        # --> Adding labels and title
plt.xlabel('Prices')
plt.ylabel('Values')
plt.title('Original VS Predicted')
plt.xticks([r + bar_width / 2 for r in range(len(categories))], categories)  # Set x-axis labels
plt.legend()
st.pyplot(grouped_bar_graph)

st.dataframe(values.describe(), width=710)