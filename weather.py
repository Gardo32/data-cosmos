import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Keys from environment
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', "2af7791e2bb94f94a8b74342241509")
NASA_API_KEY = os.environ.get('NASA_API_KEY', "VxIiaVosswm4VuinbTP2PgCOiQv8MyohW3t0EhHj")

# Weather API URLs
REALTIME_URL = os.environ.get('WEATHER_API_REALTIME_URL', 'http://api.weatherapi.com/v1/current.json')
HISTORY_URL = os.environ.get('WEATHER_API_HISTORY_URL', 'http://api.weatherapi.com/v1/history.json')
FORECAST_URL = os.environ.get('WEATHER_API_FORECAST_URL', 'http://api.weatherapi.com/v1/forecast.json')
NASA_URL = os.environ.get('NASA_API_URL', 'https://power.larc.nasa.gov/api/temporal/daily/point')

# Step 1: Fetch real-time weather data
def get_real_time_data(location):
    params = {
        'key': WEATHER_API_KEY,
        'q': location,
        'aqi': 'no'
    }
    response = requests.get(REALTIME_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Step 2: Fetch weather data for a specific date
def get_weather_data(location, date):
    params = {
        'key': WEATHER_API_KEY,
        'q': location,
        'dt': date
    }
    response = requests.get(HISTORY_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Step 3: Fetch weather forecast data
def get_forecast_data(location, days):
    params = {
        'key': WEATHER_API_KEY,
        'q': location,
        'days': days,
        'aqi': 'no'
    }
    response = requests.get(FORECAST_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Step 4: Fetch historical weather data for the past 7 days
def fetch_historical_data(location):
    historical_data = []
    today = datetime.now()
    for i in range(7):
        date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        data = get_weather_data(location, date)
        if data:
            day_data = data['forecast']['forecastday'][0]['day']
            historical_data.append({
                'Date': date,
                'Location': data['location']['name'],
                'Country': data['location']['country'],
                'Local Time': data['location']['localtime'],
                'Avg Temperature (°C)': day_data['avgtemp_c'],
                'Avg Humidity (%)': day_data['avghumidity'],
                'Total Precipitation (mm)': day_data['totalprecip_mm'],
                'Condition': day_data['condition']['text']
            })
    return historical_data

# Step 5: Fetch forecast weather data for the next 3 days
def fetch_forecast_data(location, days=3):
    data = get_forecast_data(location, days)
    forecast_data = []
    if data:
        for day in data['forecast']['forecastday']:
            forecast_data.append({
                'Date': day['date'],
                'Location': data['location']['name'],
                'Country': data['location']['country'],
                'Local Time': data['location']['localtime'],
                'Avg Temperature (°C)': day['day']['avgtemp_c'],
                'Avg Humidity (%)': day['day']['avghumidity'],
                'Total Precipitation (mm)': day['day']['totalprecip_mm'],
                'Condition': day['day']['condition']['text']
            })
    return forecast_data

# Step 6: Fetch precipitation data from NASA API
def get_precipitation_data(latitude, longitude):
    current_date = datetime.now()
    seven_days_ago = current_date - timedelta(days=7)
    three_days_ahead = current_date + timedelta(days=3)

    params = {
        'start': seven_days_ago.strftime('%Y%m%d'),
        'end': three_days_ahead.strftime('%Y%m%d'),
        'latitude': latitude,
        'longitude': longitude,
        'community': 'AG',
        'parameters': 'PRECTOTCORR',
        'format': 'JSON',
        'header': 'true',
        'api_key': NASA_API_KEY
    }

    response = requests.get(NASA_URL, params=params)
    if response.status_code == 200:
        raw_data = response.json()
        data = raw_data['properties']['parameter']['PRECTOTCORR']
        df = pd.DataFrame(list(data.items()), columns=['Date', 'Precipitation (mm)'])
        return df
    else:
        return None

# Step 7: Main function to fetch all weather data and return a combined DataFrame
def get_combined_weather_data(location, latitude, longitude):
    """Get combined weather data for a location"""
    try:
        # Fetch real-time, historical, and forecast weather data
        real_time_weather_data = get_real_time_data(location)
        historical_weather_data = fetch_historical_data(location)
        forecast_weather_data = fetch_forecast_data(location, days=3)

        # Create DataFrames for real-time, historical, and forecast data
        real_time_df = pd.DataFrame()
        if real_time_weather_data:
            current = real_time_weather_data['current']
            real_time_df = pd.DataFrame([{
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Location': real_time_weather_data['location']['name'],
                'Country': real_time_weather_data['location']['country'],
                'Local Time': real_time_weather_data['location']['localtime'],
                'Avg Temperature (°C)': current['temp_c'],
                'Avg Humidity (%)': current['humidity'],
                'Total Precipitation (mm)': current['precip_mm'],
                'Condition': current['condition']['text']
            }])

        historical_df = pd.DataFrame(historical_weather_data, columns=['Date', 'Location', 'Country', 'Local Time',
                                                                   'Avg Temperature (°C)', 'Avg Humidity (%)',
                                                                   'Total Precipitation (mm)', 'Condition'])
        forecast_df = pd.DataFrame(forecast_weather_data, columns=['Date', 'Location', 'Country', 'Local Time',
                                                               'Avg Temperature (°C)', 'Avg Humidity (%)',
                                                               'Total Precipitation (mm)', 'Condition'])

        # Combine all DataFrames
        combined_df = pd.concat([real_time_df, historical_df, forecast_df], ignore_index=True)
        
        # Add NASA precipitation data if available
        nasa_precipitation_df = get_precipitation_data(latitude, longitude)
        if nasa_precipitation_df is not None and not nasa_precipitation_df.empty:
            combined_df = pd.merge(combined_df, nasa_precipitation_df[['Date', 'Precipitation (mm)']], 
                                 on='Date', how='left')
            combined_df['Total Precipitation (mm)'] = combined_df['Precipitation (mm)'].combine_first(
                combined_df['Total Precipitation (mm)'])
            combined_df.drop(columns=['Precipitation (mm)'], inplace=True)

        # Return empty DataFrame if no data
        if combined_df.empty:
            return None, pd.DataFrame()

        # Clip precipitation values to reasonable range
        combined_df['Total Precipitation (mm)'] = combined_df['Total Precipitation (mm)'].clip(lower=0.01, upper=0.99)

        return real_time_df, combined_df

    except Exception as e:
        print(f"Error in get_combined_weather_data: {e}")
        return None, pd.DataFrame()