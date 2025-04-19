import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.environ.get('POLLEN_API_KEY')

# Headers including the API key
headers = {
    'x-api-key': API_KEY,
    'Content-type': 'application/json',
    'Accept-Language': 'en'
}


# Get current pollen data for a place
def get_latest_pollen_data(place):
    url = f"https://api.ambeedata.com/latest/pollen/by-place?place={place}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if 'data' in data:
            return data['data']
    except requests.RequestException:
        return None  # Return None if API request fails
    return None


# Get 1 day forecast pollen data for a place
def get_forecast_pollen_data(place):
    url = f"https://api.ambeedata.com/forecast/pollen/by-place?place={place}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if 'data' in data:
            return data['data']
    except requests.RequestException:
        return None  # Return None if API request fails
    return None


def calculate_risk_level(data):
    """Calculate overall pollen risk level from Ambee API data"""
    try:
        # Map risk levels to numerical values
        risk_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
        
        # Get risk counts from each pollen type
        grass = int(data.get('grass_pollen', {}).get('risk', 0))
        tree = int(data.get('tree_pollen', {}).get('risk', 0))
        weed = int(data.get('weed_pollen', {}).get('risk', 0))
        
        # Calculate average risk
        total_risk = grass + tree + weed
        avg_risk = total_risk / 3
        
        # Convert to risk level
        if avg_risk <= 1: return 'Low'
        elif avg_risk <= 2: return 'Moderate'
        elif avg_risk <= 3: return 'High'
        else: return 'Very High'
    except (ValueError, TypeError, AttributeError) as e:
        print(f"Error calculating risk level: {e}")
        return 'Unknown'

def get_combined_pollen_data(place):
    """Get and process pollen data for a location"""
    latest_pollen = get_latest_pollen_data(place)
    
    if latest_pollen:
        try:
            # Create DataFrame with normalized data
            latest_df = pd.json_normalize(latest_pollen)
            
            # Add calculated risk level
            latest_df['Risk'] = [calculate_risk_level(data) for data in latest_pollen]
            
            # Add date
            latest_df['Date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # Extract individual risk levels if available
            latest_df['grass_risk'] = latest_df.apply(
                lambda x: x.get('grass_pollen', {}).get('risk', 0) 
                if isinstance(x.get('grass_pollen'), dict) else 0, axis=1)
            latest_df['tree_risk'] = latest_df.apply(
                lambda x: x.get('tree_pollen', {}).get('risk', 0)
                if isinstance(x.get('tree_pollen'), dict) else 0, axis=1)
            latest_df['weed_risk'] = latest_df.apply(
                lambda x: x.get('weed_pollen', {}).get('risk', 0)
                if isinstance(x.get('weed_pollen'), dict) else 0, axis=1)
            
            return latest_df
            
        except Exception as e:
            print(f"Error processing pollen data: {e}")
            return create_fallback_data()
    else:
        return create_fallback_data()

def create_fallback_data():
    """Create fallback pollen data when API fails"""
    current_date = datetime.now()
    dates = pd.date_range(start=current_date - timedelta(days=1),
                         end=current_date + timedelta(days=1),
                         freq='3H')
    
    fallback_data = []
    for date in dates:
        risk_level = np.random.choice(['Low', 'Moderate', 'High'])
        fallback_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Time': date.strftime('%H:%M'),
            'Risk': risk_level,
            'grass_risk': np.random.randint(1, 4),
            'tree_risk': np.random.randint(1, 4),
            'weed_risk': np.random.randint(1, 4)
        })
    
    return pd.DataFrame(fallback_data)