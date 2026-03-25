# Weather utilities for MLB game data
import requests
from datetime import datetime
import time
import pandas as pd

# MLB Stadium locations mapping
mlb_stadiums = {
    'AZ': {'city': 'Phoenix', 'state': 'AZ', 'lat': 33.4453, 'lon': -112.0679, 'stadium': 'Chase Field'},
    'ATL': {'city': 'Atlanta', 'state': 'GA', 'lat': 33.8897, 'lon': -84.4676, 'stadium': 'Truist Park'},
    'ATH' : {'city': 'West Sacramento', 'state': 'CA', 'lat': 38.5805, 'lon': -121.5302, 'stadium': 'Georgia Baseball Stadium'},
    'BAL': {'city': 'Baltimore', 'state': 'MD', 'lat': 39.283, 'lon': -76.6218, 'stadium': 'Oriole Park'},
    'BOS': {'city': 'Boston', 'state': 'MA', 'lat': 42.3467, 'lon': -71.0972, 'stadium': 'Fenway Park'},
    'CHC': {'city': 'Chicago', 'state': 'IL', 'lat': 41.9484, 'lon': -87.6553, 'stadium': 'Wrigley Field'},
    'CWS': {'city': 'Chicago', 'state': 'IL', 'lat': 41.8299, 'lon': -87.6338, 'stadium': 'Guaranteed Rate Field'},
    'CIN': {'city': 'Cincinnati', 'state': 'OH', 'lat': 39.0974, 'lon': -84.5068, 'stadium': 'Great American Ball Park'},
    'CLE': {'city': 'Cleveland', 'state': 'OH', 'lat': 41.4962, 'lon': -81.6852, 'stadium': 'Progressive Field'},
    'COL': {'city': 'Denver', 'state': 'CO', 'lat': 39.7559, 'lon': -104.994, 'stadium': 'Coors Field'},
    'DET': {'city': 'Detroit', 'state': 'MI', 'lat': 42.339, 'lon': -83.0485, 'stadium': 'Comerica Park'},
    'HOU': {'city': 'Houston', 'state': 'TX', 'lat': 29.7571, 'lon': -95.3555, 'stadium': 'Minute Maid Park'},
    'KC': {'city': 'Kansas City', 'state': 'MO', 'lat': 39.0517, 'lon': -94.4803, 'stadium': 'Kauffman Stadium'},
    'LAA': {'city': 'Anaheim', 'state': 'CA', 'lat': 33.8003, 'lon': -117.8827, 'stadium': 'Angel Stadium'},
    'LAD': {'city': 'Los Angeles', 'state': 'CA', 'lat': 34.0739, 'lon': -118.24, 'stadium': 'Dodger Stadium'},
    'MIA': {'city': 'Miami', 'state': 'FL', 'lat': 25.7781, 'lon': -80.2195, 'stadium': 'loanDepot Park'},
    'MIL': {'city': 'Milwaukee', 'state': 'WI', 'lat': 43.028, 'lon': -87.9712, 'stadium': 'American Family Field'},
    'MIN': {'city': 'Minneapolis', 'state': 'MN', 'lat': 44.9817, 'lon': -93.2778, 'stadium': 'Target Field'},
    'NYM': {'city': 'New York', 'state': 'NY', 'lat': 40.7571, 'lon': -73.8458, 'stadium': 'Citi Field'},
    'NYY': {'city': 'New York', 'state': 'NY', 'lat': 40.8296, 'lon': -73.9262, 'stadium': 'Yankee Stadium'},
    'OAK': {'city': 'Oakland', 'state': 'CA', 'lat': 37.7516, 'lon': -122.2005, 'stadium': 'Oakland Coliseum'},
    'PHI': {'city': 'Philadelphia', 'state': 'PA', 'lat': 39.9061, 'lon': -75.1665, 'stadium': 'Citizens Bank Park'},
    'PIT': {'city': 'Pittsburgh', 'state': 'PA', 'lat': 40.4469, 'lon': -80.0057, 'stadium': 'PNC Park'},
    'SD': {'city': 'San Diego', 'state': 'CA', 'lat': 32.7073, 'lon': -117.1566, 'stadium': 'Petco Park'},
    'SF': {'city': 'San Francisco', 'state': 'CA', 'lat': 37.7786, 'lon': -122.3893, 'stadium': 'Oracle Park'},
    'SEA': {'city': 'Seattle', 'state': 'WA', 'lat': 47.5914, 'lon': -122.3326, 'stadium': 'T-Mobile Park'},
    'STL': {'city': 'St. Louis', 'state': 'MO', 'lat': 38.6226, 'lon': -90.1928, 'stadium': 'Busch Stadium'},
    'TB': {'city': 'St. Petersburg', 'state': 'FL', 'lat': 27.7682, 'lon': -82.6534, 'stadium': 'Tropicana Field'},
    'TEX': {'city': 'Arlington', 'state': 'TX', 'lat': 32.7472, 'lon': -97.0833, 'stadium': 'Globe Life Field'},
    'TOR': {'city': 'Toronto', 'state': 'ON', 'lat': 43.6414, 'lon': -79.3894, 'stadium': 'Rogers Centre'},
    'WSN': {'city': 'Washington', 'state': 'DC', 'lat': 38.8730, 'lon': -77.0074, 'stadium': 'Nationals Park'}
}

def get_weather_data(home_team, game_date, api_key):
    """
    Fetch historical weather data for a specific MLB game.
    
    Parameters:
    - home_team: MLB team abbreviation (e.g., 'NYY', 'BOS')
    - game_date: Date string in 'YYYY-MM-DD' format
    - api_key: OpenWeatherMap API key
    
    Returns:
    - Dictionary with weather data or None if error
    """
    
    if home_team not in mlb_stadiums:
        print(f"Warning: Unknown team {home_team}")
        return None
    
    stadium = mlb_stadiums[home_team]
    lat, lon = stadium['lat'], stadium['lon']
    
    # Convert date to timestamp for API call
    dt = datetime.strptime(game_date, '%Y-%m-%d')
    timestamp = int(dt.timestamp())
    
    # Visual Crossing Weather API (more reliable for historical data)
    location = f"{lat},{lon}"
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{game_date}"
    
    params = {
        'key': api_key,
        'unitGroup': 'us',  # US units (Fahrenheit, mph)
        'include': 'days',
        'elements': 'temp,humidity,precip,windspeed,winddir,pressure,cloudcover,visibility,conditions'
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            day_data = data['days'][0]
            
            return {
                'temperature': day_data.get('temp'),
                'humidity': day_data.get('humidity'),
                'precipitation': day_data.get('precip'),
                'wind_speed': day_data.get('windspeed'),
                'wind_direction': day_data.get('winddir'),
                'pressure': day_data.get('pressure'),
                'cloud_cover': day_data.get('cloudcover'),
                'visibility': day_data.get('visibility'),
                'conditions': day_data.get('conditions'),
                'stadium_city': stadium['city']
            }
        else:
            print(f"API Error {response.status_code} for {home_team} on {game_date}")
            return None
            
    except Exception as e:
        print(f"Error fetching weather for {home_team} on {game_date}: {e}")
        return None

def add_weather_to_games(game_context_df, api_key=None, delay=0.1):
    """
    Add weather data to game context dataframe.
    
    Parameters:
    - game_context_df: DataFrame with 'home_team' and 'game_date' columns
    - api_key: Weather API key (if None, adds placeholder columns)
    - delay: Seconds to wait between API calls
    
    Returns:
    - DataFrame with added weather columns
    """
    
    weather_columns = [
        'temperature', 'humidity', 'precipitation', 
        'wind_speed', 'wind_direction', 'pressure', 
        'cloud_cover', 'visibility', 'conditions'
    ]
    
    # Initialize weather columns
    for col in weather_columns:
        game_context_df[col] = None
    
    if api_key is None:
        print("No API key provided - weather columns added but not populated")
        return game_context_df
    
    successful_requests = 0
    failed_requests = 0
    
    print(f"Fetching weather data for {len(game_context_df)} games...")

    game_context_df['game_date'] = pd.to_datetime(game_context_df['game_date'])
    
    for idx, row in game_context_df.iterrows():
        home_team = row['home_team'] 
        game_date = row['game_date'].strftime('%Y-%m-%d')
        
        weather_data = get_weather_data(home_team, game_date, api_key)
        
        if weather_data:
            for col in weather_columns:
                if col in weather_data:
                    game_context_df.at[idx, col] = weather_data[col]
            successful_requests += 1
        else:
            failed_requests += 1
        
        # Rate limiting
        if successful_requests % 10 == 0:
            print(f"Processed {successful_requests + failed_requests} games...")
        
        time.sleep(delay)
    
    print(f"Weather data fetch complete: {successful_requests} successful, {failed_requests} failed")
    return game_context_df