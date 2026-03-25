from xgboost import train

from pybaseball import statcast, statcast_pitcher, statcast_pitcher_percentile_ranks, standings, statcast_batter_percentile_ranks
import pandas as pd
import numpy as np
import pybaseball
from weather_utils import add_weather_to_games
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

    
pybaseball.cache.enable()

def get_pitcher_historical_stats(pitchers):
    data = []
    for year in range(2018, 2025):
        data_pull = statcast_pitcher_percentile_ranks(year)
        data_pull = data_pull.dropna(axis=0, thresh=len(data_pull.columns) - 1).dropna(axis=1)
        data.append(data_pull)
    
    data = pd.concat(data, ignore_index=True)
    
    # Get meaningful column names (exclude metadata columns)
    meaningful_cols = [col for col in data.columns if col not in ['player_name', 'player_id', 'year']]
    
    pitch_stats = {}
    for pitcher in tqdm(pitchers):
        if pitcher in data['player_id'].unique():
            # Get latest stats for this pitcher
            latest_stats = data[data['player_id'] == pitcher].sort_values('year').iloc[-1]
            # Extract just the meaningful stat columns
            pitch_stats[pitcher] = latest_stats[meaningful_cols]
    
    # Return DataFrame with meaningful column names preserved
    result_df = pd.DataFrame.from_dict(pitch_stats, orient='index')
    result_df.columns = meaningful_cols  # Ensure column names are preserved
    return result_df

def get_batter_historical_stats(batters):
    data = []
    for year in range(2018, 2025):
        data_pull = statcast_batter_percentile_ranks(year)
        data_pull = data_pull.dropna(axis=0, thresh=len(data_pull.columns) - 1).dropna(axis=1)
        data.append(data_pull)
    
    data = pd.concat(data, ignore_index=True)
    
    # Get meaningful column names (exclude metadata columns)
    meaningful_cols = [col for col in data.columns if col not in ['player_name', 'player_id', 'year']]
    
    batter_stats = {}
    for batter in tqdm(batters):
        if batter in data['player_id'].unique():
            # Get latest stats for this batter
            latest_stats = data[data['player_id'] == batter].sort_values('year').iloc[-1]
            # Extract just the meaningful stat columns
            batter_stats[batter] = latest_stats[meaningful_cols]
    
    # Return DataFrame with meaningful column names preserved
    result_df = pd.DataFrame.from_dict(batter_stats, orient='index')
    result_df.columns = meaningful_cols  # Ensure column names are preserved
    return result_df

def get_team_records_data(games_df):
    """
    Get team records (wins/losses) up to each game date using ESPN API.
    
    Parameters:
    - games_df: DataFrame with columns ['home_team', 'away_team', 'game_date']
    
    Returns:
    - DataFrame with team records added as new columns
    """
    import requests
    import json
    
    # Get unique years from the games
    years = sorted(games_df['game_date'].dt.year.unique())
    
    # ESPN team name mapping (ESPN uses different abbreviations)
    espn_team_mapping = {
        'AZ': 'AZ', 'ATL': 'ATL', 'BAL': 'BAL', 'BOS': 'BOS', 'CHC': 'CHC',
        'CWS': 'CWS', 'CIN': 'CIN', 'CLE': 'CLE', 'COL': 'COL', 'DET': 'DET',
        'HOU': 'HOU', 'KC': 'KC', 'LAA': 'LAA', 'LAD': 'LAD', 'MIA': 'MIA',
        'MIL': 'MIL', 'MIN': 'MIN', 'NYM': 'NYM', 'NYY': 'NYY', 'OAK': 'OAK',
        'PHI': 'PHI', 'PIT': 'PIT', 'SD': 'SD', 'SF': 'SF', 'SEA': 'SEA',
        'STL': 'STL', 'TB': 'TB', 'TEX': 'TEX', 'TOR': 'TOR', 'WSH': 'WSH', 'ATH' : 'ATH',
    }
    
    def get_standings_from_espn(year):
        """Get MLB standings from ESPN API for a given year"""
        try:
            # ESPN MLB standings API endpoint
            url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
            params = {
                'dates': f"{year}0401-{year}1031",  # April to October
                'seasontype': 2  # Regular season
            }
            
            # Try to get current season standings
            standings_url = f"https://site.api.espn.com/apis/v2/sports/baseball/mlb/standings?season={year}"
            
            response = requests.get(standings_url, timeout=10)
            response.raise_for_status()
            
            standings_data = response.json()
            
            # Parse the standings data
            teams_records = {}
            
            if 'children' in standings_data:
                # Navigate through the ESPN API structure
                for conference in standings_data['children']:  # AL/NL
                    if 'standings' in conference:
                        for division in conference['standings']['entries']:
                            team = division['team']
                            stats = division['stats']
                            
                            # Extract team abbreviation and record
                            team_abbr = team['abbreviation']
                            
                            # Find wins and losses in stats
                            wins = losses = 0
                            for stat in stats:
                                if stat['name'] == 'wins':
                                    wins = stat['value']
                                elif stat['name'] == 'losses':
                                    losses = stat['value']
                            
                            teams_records[team_abbr] = {
                                'wins': wins,
                                'losses': losses,
                                'win_pct': wins / (wins + losses) if (wins + losses) > 0 else 0.5
                            }
            
            if teams_records:
                print(f"✅ Loaded {len(teams_records)} teams from ESPN for {year}")
                return teams_records
            else:
                print(f"⚠️ No standings data found for {year}")
                return {}
                
        except Exception as e:
            print(f"❌ Error loading standings for {year}: {e}")
            return {}
    
    # Pull standings for all years
    all_standings = {}
    for year in tqdm(years, desc="Loading standings from ESPN"):
        year_standings = get_standings_from_espn(year)
        if year_standings:
            all_standings[year] = year_standings
    
    if not all_standings:
        print("ℹ️ No standings data available, using default values (0.500 win%)")
        games_df['home_team_wins'] = 0
        games_df['home_team_losses'] = 0
        games_df['home_team_win_pct'] = 0.5
        games_df['away_team_wins'] = 0
        games_df['away_team_losses'] = 0
        games_df['away_team_win_pct'] = 0.5
        return games_df
    
    def get_team_record_on_date(team, date, all_standings):
        """Get team's record for the season (ESPN provides season totals)"""
        year = date.year
        
        # Map team name to ESPN format if needed
        espn_team = espn_team_mapping.get(team, team)
        
        if year not in all_standings:
            return 0, 0, 0.5
            
        year_standings = all_standings[year]
        
        # Try both original and mapped team names
        for team_name in [team, espn_team]:
            if team_name in year_standings:
                record = year_standings[team_name]
                return record['wins'], record['losses'], record['win_pct']
        
        # Default if team not found
        return 0, 0, 0.5
    
    # Add team record columns
    print("Adding team records to game context...")
    
    # Initialize columns
    games_df['home_team_wins'] = 0
    games_df['home_team_losses'] = 0
    games_df['home_team_win_pct'] = 0.5
    games_df['away_team_wins'] = 0
    games_df['away_team_losses'] = 0
    games_df['away_team_win_pct'] = 0.5
    
    # Fill in records for each game
    for idx, row in tqdm(games_df.iterrows(), total=len(games_df), desc="Processing team records"):
        # Home team record
        home_wins, home_losses, home_win_pct = get_team_record_on_date(
            row['home_team'], row['game_date'], all_standings
        )
        games_df.loc[idx, 'home_team_wins'] = home_wins
        games_df.loc[idx, 'home_team_losses'] = home_losses
        games_df.loc[idx, 'home_team_win_pct'] = home_win_pct
        
        # Away team record
        away_wins, away_losses, away_win_pct = get_team_record_on_date(
            row['away_team'], row['game_date'], all_standings
        )
        games_df.loc[idx, 'away_team_wins'] = away_wins
        games_df.loc[idx, 'away_team_losses'] = away_losses
        games_df.loc[idx, 'away_team_win_pct'] = away_win_pct
    
    return games_df

def get_transformed_data(start_dt, end_dt, weather = True, weather_api_key='VQVKFHQS24N3XBA9HAY2TJZWK'):
    """
    Transform Statcast data with optional weather information.

    Parameters:
    - start_dt, end_dt: Date range for Statcast data
    - weather_api_key: Optional API key for weather data (Visual Crossing Weather)
                        If None, weather columns are added but not populated
                        Get free key at: https://www.visualcrossing.com/weather-api
    """
    # Import weather utilities
    df = statcast(start_dt = start_dt, end_dt = end_dt).sort_values(by=['game_date', 'at_bat_number'])
    minimal_pitch_columns = [
        'pitch_type',           # What type of pitch
        'release_speed',        # How fast
        'plate_x',              # Horizontal location at plate
        'plate_z',              # Vertical location at plate
        'pfx_x',                # Horizontal movement 
        'pfx_z',                # Vertical movement
        'release_spin_rate',    # Spin rate
        'zone'                  # Strike zone location
        ]

    pitch_context_cols = [
    # Count situation - heavily influences pitch selection
    'balls',                    # Pre-pitch ball count
    'strikes',                  # Pre-pitch strike count

    # Game situation
    'inning',                   # Which inning
    'inning_topbot',            # Top or bottom of inning
    'outs_when_up',             # Number of outs
    'home_score',               # Home team score
    'away_score',               # Away team score
    'bat_score_diff',           # Batting team score differential

    # Baserunner situation - affects pitch strategy
    'on_1b',                    # Runner on 1st base
    'on_2b',                    # Runner on 2nd base  
    'on_3b',                    # Runner on 3rd base

    # Player characteristics
    'stand',                    # Batter handedness (L/R)
    'p_throws',                 # Pitcher handedness (L/R)
    'batter',                   # Batter ID (for player-specific tendencies)
    'pitcher',                  # Pitcher ID (for pitcher-specific tendencies)

    # Pitch sequence context
    'pitch_number',             # Pitch number in at-bat
    'at_bat_number',            # At-bat number in game

    # Strike zone reference
    'sz_top',                   # Top of batter's strike zone
    'sz_bot',                   # Bottom of batter's strike zone

    # Pitcher workload/fatigue indicators  
    'n_thruorder_pitcher',      # Times through batting order for pitcher
    'pitcher_days_since_prev_game',  # Days rest for pitcher

    # Batter recent activity
    'n_priorpa_thisgame_player_at_bat',  # Prior plate appearances this game
    'batter_days_since_prev_game',       # Days since batter's last game

    # Age factors (experience)
    'age_pit',                  # Pitcher age
    'age_bat',                  # Batter age

    # Defensive positioning
    'if_fielding_alignment',    # Infield alignment
    'of_fielding_alignment',    # Outfield alignment
    'home_team',
    'away_team',
    'game_date',
    'at_bat_number',
    ]

    game_context_cols = [
    'game_type',                # Regular season, playoffs, etc.
    'home_team',                # Home team
    'away_team',                # Away team
    'game_date',                # Date of game
    ]

    # weather = False
    # Create base game context
    game_context = (
    df.sort_values(['home_team', 'game_date', 'at_bat_number'])
        .groupby(['home_team', 'game_date'])[game_context_cols]
        .first()
    ).reset_index(drop=True)

    # Ensure game_date is datetime format for team records processing
    game_context['game_date'] = pd.to_datetime(game_context['game_date'])

    # Add weather data to game context
    # if weather:
        # print(f"Adding weather data to {len(game_context)} games...")
        # game_context = add_weather_to_games(game_context, weather_api_key)

    pitch_context = df[pitch_context_cols]
    pitch = df[minimal_pitch_columns]
    play_results = df['events']
    pitch_results = df['description']
    home_team_win_target = (
        df.sort_values(['home_team', 'game_date', 'at_bat_number'])
        .groupby(['home_team', 'game_date'])['home_win_exp']
        .last()
        .pipe(np.round)
        )

    # Get pitcher historical stats
    pitcher_stats = get_pitcher_historical_stats(pitchers=df['pitcher'].unique())

    batter_stats = get_batter_historical_stats(batters=df['batter'].unique())

    # Create pitcher mapping for each game
    pitcher_by_game = (
        df.sort_values(['home_team', 'game_date', 'at_bat_number'])
        .groupby(['home_team', 'game_date'])['pitcher']
        .first()  # Get the starting pitcher for each game
        .reset_index()
    )

    # Ensure both DataFrames have proper datetime format before merge
    pitcher_by_game['game_date'] = pd.to_datetime(pitcher_by_game['game_date'])
    game_context['game_date'] = pd.to_datetime(game_context['game_date'])

    # Merge game context with pitcher info
    game_context = game_context.merge(
        pitcher_by_game,
        on=['home_team', 'game_date'],
        how='left'
    )

    # Add pitcher stats columns to game context
    if not pitcher_stats.empty:
        # Get the column names from pitcher stats (excluding pitcher info columns)
        pitcher_stat_cols = pitcher_stats.columns.tolist()
        
        # Create a default row of zeros for pitchers not in historical data
        default_stats = pd.Series(0.0, index=pitcher_stat_cols)
        
        # Map each pitcher to their stats, using defaults if not found
        for col in pitcher_stat_cols:
            game_context[f'pitcher_{col}'] = game_context['pitcher'].map(
                lambda p: pitcher_stats.loc[p, col] if p in pitcher_stats.index else default_stats[col]
            )

    # Drop the temporary pitcher column
    game_context.drop(columns=['pitcher'], inplace=True)

    # Add team records data
    game_context = get_team_records_data(game_context)

    pitch_context = pitch_context.merge(pitcher_stats, left_on='pitcher', right_index=True, how='left', suffixes=('', '_pit'))
    pitch_context = pitch_context.merge(batter_stats, left_on='batter', right_index=True, how='left', suffixes=('', '_bat'))

    pitch_context = pitch_context.drop(columns=['age_pit', 'age_bat', 'if_fielding_alignment', 'of_fielding_alignment', 'oaa'])
    pitch_context[['on_1b', 'on_2b', 'on_3b']] = pitch_context[['on_1b', 'on_2b', 'on_3b']].fillna(0)

    pitch_context = pitch_context.loc[:, ~pitch_context.columns.duplicated()].copy()
    pitch_context.index = (pitch_context['game_date'] + '_' + pitch_context['home_team'] + '_' + pitch_context['away_team'] + '_' + pitch_context['at_bat_number'].astype(str) + '_' + pitch_context['pitch_number'].astype(str))
    pitch_context.drop(columns=['game_date'], inplace=True)
    pitch_context = pd.get_dummies(pitch_context, columns=[col for col in pitch_context.select_dtypes(include=['object']).columns if col != 'game_date'], drop_first=True).astype(float)
    pitch_context = pitch_context.fillna(pitch_context.mean())

    pitch.index = pitch_context.index
    game_context.index = game_context['game_date'].astype(str) + '_' + game_context['home_team'] + '_' + game_context['away_team']
    play_results.index = pitch_context.index
    pitch_results.index = pitch_context.index
    home_team_win_target.index = game_context.index


    pitch_context['game_id'] = pitch_context.index.str.split('_').str[0:3].str.join('_')
    pitch_context['game_date'] = pitch_context.index.str.split('_').str[0]
    pitch_context['home_team'] = pitch_context.index.str.split('_').str[1]
    pitch_context['away_team'] = pitch_context.index.str.split('_').str[2]
    pitch_context['at_bat_id'] = pitch_context.index.str.split('_').str[3]
    pitch_context['pitch_id'] = pitch_context.index.str.split('_').str[4]

    pitch['game_id'] = pitch_context['game_id']
    pitch['game_date'] = pitch_context['game_date']
    pitch['home_team'] = pitch_context['home_team']
    pitch['away_team'] = pitch_context['away_team']
    pitch['at_bat_id'] = pitch_context['at_bat_id']
    pitch['pitch_id'] = pitch_context['pitch_id']

    pitch = pitch.sort_values(['at_bat_id', 'pitch_id'])

    pitch_context = pitch_context.sort_values(['at_bat_id', 'pitch_id'])
    batter_context = pitch_context.groupby('game_id', as_index=False).first()[['game_id', 'batter_days_since_prev_game', 'xwoba', 'xba','xslg', 'xiso', 'xobp', 'brl', 'brl_percent', 'exit_velocity', 'max_ev', 'hard_hit_percent',
                   'k_percent', 'bb_percent', 'whiff_percent', 'chase_percent', 'xera', 'fb_velocity', 'fb_spin', 'curve_spin', 'xwoba_bat', 'xba_bat',
                   'xslg_bat', 'xiso_bat', 'xobp_bat', 'brl_bat', 'brl_percent_bat', 'exit_velocity_bat', 'max_ev_bat', 'hard_hit_percent_bat',
                   'k_percent_bat', 'bb_percent_bat', 'whiff_percent_bat', 'chase_percent_bat', 'arm_strength', 'sprint_speed', 'bat_speed', 'squared_up_rate',
                   'swing_length', 'inning_topbot_Top', 'stand_R', 'p_throws_R']]
    
    game_context = game_context.merge(batter_context, left_index = True, right_on='game_id', how='left')


    res_dict = {
        'game_context': game_context,
        'pitch_context': pitch_context,
        'pitch' : pitch,
        'first_pitch' : pitch.groupby('game_id').first().drop(columns=['game_date', 'home_team', 'away_team', 'at_bat_id', 'pitch_id']),
        'pitch_result' : pitch_results,
        'at_bat_target' : play_results,
        'Game_target' : home_team_win_target.reset_index()['home_win_exp']
    }   

    return res_dict