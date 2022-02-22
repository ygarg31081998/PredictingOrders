from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from random import randint
import random
import flask

app = Flask(__name__)

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def moving_average_forecast(series, window_size):
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean(axis = 0))
  return np.array(forecast)

def predict_sales(series, seasonality=12, window_size=24, duration=12):
    dataset_length = len(series)
    if(dataset_length <= seasonality):
        meanValue = series.mean(axis = 0)
        return np.full((duration, 1), meanValue, dtype=int)
       
    diff_series = (series[seasonality:] - series[:-seasonality])
    diff_moving_avg = moving_average_forecast(diff_series, window_size)[dataset_length - seasonality - window_size-duration:]
    print(diff_moving_avg.shape)
    diff_moving_avg_plus_past = series[dataset_length - duration - seasonality:-seasonality] + diff_moving_avg
    return diff_moving_avg_plus_past


size = 180
products = 5
time = np.arange(size, dtype="float32")
series = np.arange(size * products,dtype="float32").reshape(size, products)
p1 = trend(time, random.random())
p2 = trend(time, random.random())
p3 = trend(time, random.random())
p4 = trend(time, random.random())
p5= trend(time, random.random())  


# Create the series
p1 = randint(40, 100) + trend(time, random.random()) + seasonality(time, period=12, amplitude=randint(20, 50))
p2 = randint(40, 100) + trend(time, random.random()) + seasonality(time, period=12, amplitude=randint(20, 50))
p3 = randint(40, 100) + trend(time, random.random()) + seasonality(time, period=12, amplitude=randint(20, 50))
p4 = randint(40, 100) + trend(time, random.random()) + seasonality(time, period=12, amplitude=randint(20, 50))
p5 = randint(40, 100) + trend(time, random.random()) + seasonality(time, period=12, amplitude=randint(20, 50))
# Update with noise
p1 += noise(time, randint(1,5), seed=42)
p2 += noise(time, randint(1,5), seed=42)
p3 += noise(time, randint(1,5), seed=42)
p4 += noise(time, randint(1,5), seed=42)
p5 += noise(time, randint(1,5), seed=42)

series = np.column_stack((p1,p2,p3,p4,p5))

# 80% for train data
split_time = int(size * 0.8)
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
window_size = 12
  
@app.route('/')
def index():
    return "hello world"


@app.route('/predict')
def predict():
    result = predict_sales(series).astype(int)
    print(result)
    return str(result)


if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='localhost', port=8081)