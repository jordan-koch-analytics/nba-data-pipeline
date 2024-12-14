import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.static import teams, players
import psycopg2
from psycopg2 import extras
import os

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Endpoint Functions
# ==================
def get_game_logs(start_date_str, end_date_str, max_retries=3, initial_wait=2):
    """
    Fetches game logs over a specified period from nba-api.
    
    Paramters:
    - start_date (str): The first date in the range
    - end_date (str): The last day in the range
    - max_retries (int): Maximum number of retry attempts.
    - initial_wait (int): Initial wait time (in seconds) before the first retry.
    
    Returns:
    - pd.DataFrame: A Pandas dataframe containing game logs from the specified period.
    """
    
    # Initialize the NBA Game Finder
    gamefinder = leaguegamefinder.LeagueGameFinder(
        league_id_nullable='00',            # '00' corresponds to the NBA
        season_nullable='2024-25',          # Get current season
        date_from_nullable=start_date_str,
        date_to_nullable=end_date_str
    )
    
    for attempt in range(max_retries):
        try:
            game_logs_df = gamefinder.get_data_frames()[0]
            
            return game_logs_df
        
        except Exception as e:
            logger.error(f'Error in get_game_logs: {str(e)}')
            
            # Retry until max retries
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # All attempts have failed
                record_game_log_failure(start_date_str, end_date_str, 3, str(e))
                raise RuntimeError(f"Failed to fetch game logs after {max_retries} attempts.")

def process_game_logs(df):
    """
    Processes raw boxscore data and returns the processed data as a Pandas dataframe.
    
    Paramters:
    - pd.DataFrame: a Pandas dataframe containing raw box score data.
    
    Returns:
    - pd.DataFrame: A Pandas dataframe containing processed player data.
    """
    
    try:
        # Fix column structure
        processed_df = df.drop(['TEAM_ABBREVIATION', 'TEAM_NAME', 'FG_PCT', 'FT_PCT', 'FG3_PCT'], axis=1, errors='ignore')    # Drop unneccary columns
        processed_df.columns = processed_df.columns.str.lower()    # Lowercase columns
        
        # Fix datatypes
        processed_df['season_id'] = pd.to_numeric(processed_df['season_id'], errors='coerce').fillna(0).astype('int64')
        processed_df['game_date'] = pd.to_datetime(processed_df['game_date'])
        processed_df['season_id'] = pd.to_numeric(processed_df['season_id'], errors='coerce').fillna(0).astype('int64')
        
        return processed_df
    
    except Exception as e:
        logger.error(f'Error in process_game_logs: {str(e)}')
        raise e

def get_boxscores(game_id, max_retries=3, initial_wait=2):
    """
    Fetches box score data for a specific game.
    
    Parameters:
    - game_id (str): The unique identifier for the game.
    - max_retries (int): Maximum number of retry attempts.
    - initial_wait (int): Initial wait time in seconds before the first retry.
    
    Returns:
    - pd.DataFrame: A Pandas DataFrame containing the boxscore for the game.
                    Returns an empty DataFrame if no data or if all attempts fail.
    """
    for attempt in range(max_retries):
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            boxscore_df = boxscore.get_data_frames()[0]
            logger.info(f"Fetched box scores for game_id: {game_id} on attempt {attempt+1}")
            return boxscore_df
        except Exception as e:
            logger.error(f"Error fetching box scores for game_id {game_id} (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying get_boxscores in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                record_boxscore_failure(game_id, 3, str(e))
                logger.error(f"Failed to fetch box scores for game_id {game_id} after {max_retries} attempts.")
                return pd.DataFrame()

def process_boxscores(boxscores_df):
    """
    Processes boxscore data to prepare for RDS insertion.
    
    Parameters:
    - boxscores_df (pd.DataFrame): A Pandas DataFrame containing boxscore data.
    
    Returns:
    - pd.DataFrame: A Pandas DataFrame containing processed boxscore data.
    """
    try:
        clean_boxscores_df = boxscores_df.drop(
            ['TEAM_ABBREVIATION', 'TEAM_CITY', 'PLAYER_NAME', 'NICKNAME', 'FG_PCT', 'FG3_PCT', 'FT_PCT'],
            axis=1, errors='ignore'
        )
        clean_boxscores_df.columns = clean_boxscores_df.columns.str.lower()

        # Convert columns to Int64
        cols_to_convert_to_int64 = [
            'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
            'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk',
            'to', 'pf', 'pts', 'plus_minus'
        ]
        for col in cols_to_convert_to_int64:
            if col in clean_boxscores_df.columns:
                clean_boxscores_df[col] = clean_boxscores_df[col].astype('Int64')

        # Convert "min" column to float if it exists
        if 'min' in clean_boxscores_df.columns:
            clean_boxscores_df['min'] = clean_boxscores_df['min'].replace('None', np.nan)
            min_split = clean_boxscores_df['min'].astype(str).str.split(':', expand=True)
            clean_boxscores_df['minutes'] = pd.to_numeric(min_split[0], errors='coerce')
            clean_boxscores_df['seconds'] = pd.to_numeric(min_split[1], errors='coerce')
            clean_boxscores_df['min_float'] = (clean_boxscores_df['minutes'] + (clean_boxscores_df['seconds'] / 60)).round(2)
            clean_boxscores_df['min'] = clean_boxscores_df['min_float']
            clean_boxscores_df.drop(['minutes', 'seconds', 'min_float'], axis=1, errors='ignore', inplace=True)
        else:
            logger.warning("No 'min' column in boxscores. Setting 'min' to null.")
            clean_boxscores_df['min'] = pd.NA
            
        # Convert Int64 columns to Python ints or None
        for col in cols_to_convert_to_int64:
            if col in clean_boxscores_df.columns:
                clean_boxscores_df[col] = clean_boxscores_df[col].apply(lambda x: int(x) if pd.notnull(x) else None)

        # Convert 'min' to float or None
        if 'min' in clean_boxscores_df.columns:
            clean_boxscores_df['min'] = clean_boxscores_df['min'].apply(lambda x: float(x) if pd.notnull(x) else None)

        for col in ['team_id', 'player_id']:
            if col in clean_boxscores_df.columns and str(clean_boxscores_df[col].dtype) == 'Int64':
                clean_boxscores_df[col] = clean_boxscores_df[col].apply(lambda x: int(x) if pd.notnull(x) else None)
                
        # Rename columns
        clean_boxscores_df.rename(columns={
            'to': 'turnovers'
        }, inplace=True)
            
        return clean_boxscores_df
    
    except Exception as e:
        logger.error(f'Error in process_boxscores: {str(e)}')
        raise e



# Static Data Functions
# =====================
def get_teams():
    """
    Fetches static team data from nba_api and returns it as a DataFrame.
    
    Parameters:
    - None
    
    Returns:
    - pd.DataFrame: Pandas dataframe comtaining team data.
    """
    
    try:
        # Fetch teams data
        nba_teams = teams.get_teams()
        teams_df = pd.DataFrame(nba_teams)
        
        # Process data
        teams_df = process_teams(teams_df)
        logger.info(f"Fetched and processed {len(teams_df)} teams from nba_api")
        
        return teams_df
    
    except Exception as e:
        logger.error(f"Error in fetch_and_store_teams: {str(e)}")
        raise e

def process_teams(teams_df):
    """
    Processes static team data and returns the processed data as a dataframe.
    
    Parameters:
    - teams_df (pd.DataFrame): A Pandas dataframe containing raw team data.
    
    Returns:
    - pd.DataFrame: A Pandas dataframe contaning processed team data.
    """
    
    try:
        required_cols = {'id', 'full_name', 'nickname', 'abbreviation', 'city', 'state', 'year_founded'}
        missing = required_cols - set(teams_df.columns)
        if missing:
            logger.warning(f"Missing expected columns: {missing}")
        
        processed_df = teams_df[['id', 'full_name', 'nickname', 'abbreviation', 'city', 'state', 'year_founded']]
        processed_df.rename(columns={
            'id': 'team_id',
            'abbreviation': 'team_abbreviation',
            'full_name': 'team_name'
        }, inplace=True)
        
        # Logging
        logger.info(f'Processed team data.')
        
        return processed_df
    
    except Exception as e:
        logger.error(f'Error in process_teams: {str(e)}')
        raise e

def get_players():
    """
    Fetches static player data from nba_api and returns it as a Pandas dataframe.
    
    Parameters:
    - None
    
    Returns:
    - pd.DataFrame: a Pandass dataframe containing player data.
    """
    
    try:
        # Fetch players data
        nba_players = players.get_players()
        players_df = pd.DataFrame(nba_players)
        
        # Process player data
        processed_players_df = process_players(players_df)
        logger.info(f"Fetched {len(players_df)} players from nba_api")
        
        return processed_players_df
    
    except Exception as e:
        logger.error(f"Error in fetch_and_store_players: {str(e)}")
        raise e

def process_players(players_df):
    """
    Processes static player data and returns the processed data as a Pandas dataframe.
    
    Paramters:
    - players_df (pd.DataFrame): A Pandas dataframe containing raw player data.
    
    Returns:
    - pd.DataFrame: A Pandas dataframe containing processed player data.
    """
    
    try:
        required_cols = {'is_active', 'id', 'full_name'}
        missing = required_cols - set(players_df.columns)
        if missing:
            logger.warning(f"Missing expected columns: {missing}")

        df_processed = players_df[players_df['is_active'] == True]    # Get active players
        df_processed.drop('is_active', axis=1, errors='ignore', inplace=True)
        
        df_processed.rename(columns={
            'id': 'player_id',
            'full_name': 'player_name'
        }, inplace=True)
        
        # Logging
        logger.info('Processed player data.')
        
        return df_processed
    
    except Exception as e:
        logger.error(f'Error in process_players: {str(e)}')
        raise e


        
        
        
        

# Database Functions
# ==================        
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise e

def read_sql_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def store_teams_in_rds(teams_df):
    """
    Upsert teams data into the 'teams' table in RDS.
    
    teams_df columns:
    - team_id (int64)
    - team_name (object)
    - nickname (object)
    - team_abbreviation (object)
    - city (object)
    - state (object)
    - year_founded (int64)
    
    Paramters:
        - teams_df (pd.DataFrame): A Pandas dataframe containing processed data from the NBA API.
        
    Returns:
        - None
    """
    if teams_df.empty:
        logger.info("No team data to store.")
        return
    
    insert_query = """
        INSERT INTO teams (team_id, team_name, nickname, team_abbreviation, city, state, year_founded)
        VALUES %s
        ON CONFLICT (team_id)
        DO UPDATE SET
          team_name = EXCLUDED.team_name,
          nickname = EXCLUDED.nickname,
          team_abbreviation = EXCLUDED.team_abbreviation,
          city = EXCLUDED.city,
          state = EXCLUDED.state,
          year_founded = EXCLUDED.year_founded;
    """

    records = []
    for row in teams_df.itertuples(index=False):
        record = (
            row.team_id,
            row.team_name,
            row.nickname,
            row.team_abbreviation,
            row.city,
            row.state,
            row.year_founded
        )
        records.append(record)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        extras.execute_values(cursor, insert_query, records, page_size=100)
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Upserted {len(records)} teams into RDS successfully.")
    
    except Exception as e:
        logger.error(f"Failed to store teams in RDS: {str(e)}")
        raise e

def store_players_in_rds(players_df):
    """
    Upsert players data into the 'players' table in RDS.
    
    players_df columns:
    - player_id (int64)
    - player_name (object)
    - first_name (object)
    - last_name (object)
    
    Parameters:
        - players_df (pd.DataFrame): A Pandas dataframe containing processed player data from the NBA API.
        
    Returns:
        - None
    """
    if players_df.empty:
        logger.info("No player data to store.")
        return
    
    insert_query = """
        INSERT INTO players (player_id, player_name, first_name, last_name)
        VALUES %s
        ON CONFLICT (player_id)
        DO UPDATE SET
          player_name = EXCLUDED.player_name,
          first_name = EXCLUDED.first_name,
          last_name = EXCLUDED.last_name;
    """

    records = []
    for row in players_df.itertuples(index=False):
        record = (
            row.player_id,
            row.player_name,
            getattr(row, 'first_name', None),
            getattr(row, 'last_name', None)
        )
        records.append(record)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        extras.execute_values(cursor, insert_query, records, page_size=100)
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Upserted {len(records)} players into RDS successfully.")
        
    except Exception as e:
        logger.error(f"Failed to store players in RDS: {str(e)}")
        raise e
        
def store_games_in_rds(game_logs_df):
    """
    Upsert game logs data into the 'game_logs' table in RDS.
    
    game_logs_df columns:
    - game_id (object)
    - team_id (int64)
    - season_id (int64)
    - game_date (datetime64[ns])
    - matchup (object)
    - wl (object)
    - min (int64)
    - pts (int64)
    - fgm (int64)
    - fga (int64)
    - fg3m (int64)
    - fg3a (int64)
    - ftm (int64)
    - fta (int64)
    - oreb (int64)
    - dreb (int64)
    - reb (int64)
    - ast (int64)
    - stl (int64)
    - blk (int64)
    - tov (int64)
    - pf (int64)
    - plus_minus (float64)
    
    Primary Key: (game_id, team_id)
    
    Parameters:
        - game_logs_df (pd.DataFrame): A Pandas dataframe containing processed 
          game log data from the NBA API.
          
    Returns:
        - None
    """

    if game_logs_df.empty:
        logger.info("No game log data to store.")
        return

    columns = [
        'game_id',
        'team_id',
        'season_id',
        'game_date',
        'matchup',
        'wl',
        'min',
        'pts',
        'fgm',
        'fga',
        'fg3m',
        'fg3a',
        'ftm',
        'fta',
        'oreb',
        'dreb',
        'reb',
        'ast',
        'stl',
        'blk',
        'tov',
        'pf',
        'plus_minus'
    ]

    # Construct the insert query with ON CONFLICT
    insert_query = f"""
        INSERT INTO game_logs ({", ".join(columns)})
        VALUES %s
        ON CONFLICT (game_id, team_id)
        DO UPDATE SET
          season_id = EXCLUDED.season_id,
          game_date = EXCLUDED.game_date,
          matchup = EXCLUDED.matchup,
          wl = EXCLUDED.wl,
          min = EXCLUDED.min,
          pts = EXCLUDED.pts,
          fgm = EXCLUDED.fgm,
          fga = EXCLUDED.fga,
          fg3m = EXCLUDED.fg3m,
          fg3a = EXCLUDED.fg3a,
          ftm = EXCLUDED.ftm,
          fta = EXCLUDED.fta,
          oreb = EXCLUDED.oreb,
          dreb = EXCLUDED.dreb,
          reb = EXCLUDED.reb,
          ast = EXCLUDED.ast,
          stl = EXCLUDED.stl,
          blk = EXCLUDED.blk,
          tov = EXCLUDED.tov,
          pf = EXCLUDED.pf,
          plus_minus = EXCLUDED.plus_minus;
    """

    # Convert DataFrame rows to tuples for execute_values
    records = []
    for row in game_logs_df.itertuples(index=False):
        record = (
            getattr(row, 'game_id'),
            getattr(row, 'team_id'),
            getattr(row, 'season_id'),
            getattr(row, 'game_date').to_pydatetime() if hasattr(getattr(row, 'game_date'), 'to_pydatetime') else getattr(row, 'game_date'),
            getattr(row, 'matchup'),
            getattr(row, 'wl'),
            getattr(row, 'min'),
            getattr(row, 'pts'),
            getattr(row, 'fgm'),
            getattr(row, 'fga'),
            getattr(row, 'fg3m'),
            getattr(row, 'fg3a'),
            getattr(row, 'ftm'),
            getattr(row, 'fta'),
            getattr(row, 'oreb'),
            getattr(row, 'dreb'),
            getattr(row, 'reb'),
            getattr(row, 'ast'),
            getattr(row, 'stl'),
            getattr(row, 'blk'),
            getattr(row, 'tov'),
            getattr(row, 'pf'),
            float(getattr(row, 'plus_minus')) if pd.notnull(getattr(row, 'plus_minus')) else None
        )
        records.append(record)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        extras.execute_values(cursor, insert_query, records, page_size=100)
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Upserted {len(records)} game log records into RDS successfully.")
        
    except Exception as e:
        logger.error(f"Failed to store game logs in RDS: {str(e)}")
        raise e
        
def store_boxscores_in_rds(boxscores_df):
    """
    Upsert boxscore data into the 'boxscores' table in RDS.

    Columns in boxscores_df:
    - game_id (object/string)
    - team_id (int64)
    - player_id (int64)
    - start_position (object)
    - comment (object)
    - min (float64)
    - fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, turnovers, pf, pts, plus_minus (Int64)

    Primary Key: (game_id, player_id)
    
    Parameters:
        - boxscores_df (pd.DataFrame): A Pandas dataframe containing process box score data.
        
    Returns:
        - None
    """

    if boxscores_df.empty:
        logger.info("No boxscore data to store.")
        return

    columns = [
        'game_id',
        'team_id',
        'player_id',
        'start_position',
        'comment',
        'min',
        'fgm',
        'fga',
        'fg3m',
        'fg3a',
        'ftm',
        'fta',
        'oreb',
        'dreb',
        'reb',
        'ast',
        'stl',
        'blk',
        'turnovers',
        'pf',
        'pts',
        'plus_minus'
    ]

    insert_query = f"""
        INSERT INTO boxscores ({", ".join(columns)})
        VALUES %s
        ON CONFLICT (game_id, player_id)
        DO UPDATE SET
          team_id = EXCLUDED.team_id,
          start_position = EXCLUDED.start_position,
          comment = EXCLUDED.comment,
          min = EXCLUDED.min,
          fgm = EXCLUDED.fgm,
          fga = EXCLUDED.fga,
          fg3m = EXCLUDED.fg3m,
          fg3a = EXCLUDED.fg3a,
          ftm = EXCLUDED.ftm,
          fta = EXCLUDED.fta,
          oreb = EXCLUDED.oreb,
          dreb = EXCLUDED.dreb,
          reb = EXCLUDED.reb,
          ast = EXCLUDED.ast,
          stl = EXCLUDED.stl,
          blk = EXCLUDED.blk,
          turnovers = EXCLUDED.turnovers,
          pf = EXCLUDED.pf,
          pts = EXCLUDED.pts,
          plus_minus = EXCLUDED.plus_minus;
    """

    records = []
    for row in boxscores_df.itertuples(index=False):
        # Helper function to convert possibly pd.NA or np.int64 to Python int or None
        def safe_int(x):
            if pd.isnull(x):
                return None
            # Convert numpy.int64 to int
            return int(x) if isinstance(x, (np.integer, np.int64)) else x

        def safe_float(x):
            if pd.isnull(x):
                return None
            # Convert if needed
            return float(x)

        record = (
            getattr(row, 'game_id'),
            safe_int(getattr(row, 'team_id')),
            safe_int(getattr(row, 'player_id')),
            getattr(row, 'start_position', None),
            getattr(row, 'comment', None),
            safe_float(getattr(row, 'min')),  # min is float
            safe_int(getattr(row, 'fgm', None)),
            safe_int(getattr(row, 'fga', None)),
            safe_int(getattr(row, 'fg3m', None)),
            safe_int(getattr(row, 'fg3a', None)),
            safe_int(getattr(row, 'ftm', None)),
            safe_int(getattr(row, 'fta', None)),
            safe_int(getattr(row, 'oreb', None)),
            safe_int(getattr(row, 'dreb', None)),
            safe_int(getattr(row, 'reb', None)),
            safe_int(getattr(row, 'ast', None)),
            safe_int(getattr(row, 'stl', None)),
            safe_int(getattr(row, 'blk', None)),
            safe_int(getattr(row, 'turnovers', None)),
            safe_int(getattr(row, 'pf', None)),
            safe_int(getattr(row, 'pts', None)),
            safe_int(getattr(row, 'plus_minus', None))
        )
        records.append(record)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        extras.execute_values(cursor, insert_query, records, page_size=100)
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Upserted {len(records)} boxscore records into RDS successfully.")
    except Exception as e:
        logger.error(f"Failed to store boxscores in RDS: {str(e)}")
        raise e
        
def record_game_log_failure(start_date_str, end_date_str, attempt_count, error_message):
    """
    Stores failures while fetching game logs in the game_log_failures table in RDS.
    
    Parameters:
        - start_date_str (str): A string representing the start date for the game logs. ('%m/%d/%Y')
        - end_date_str (str): A string representing the end date for the game logs. ('%m/%d/%Y')
        - attempt_count (int): The number of unsuccessful attepmts to retrieve the game logs.
        - error_message (str): A string containing the exception thrown while fetching game logs.
        
    Returns:
        - None
    """
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO game_log_failures (start_date_str, end_date_str, attempt_count, error_message, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (start_date_str, end_date_str)
            DO UPDATE SET
              attempt_count = game_log_failures.attempt_count + EXCLUDED.attempt_count,
              error_message = EXCLUDED.error_message,
              created_at = NOW();
        """
        cursor.execute(insert_query, (start_date_str, end_date_str, attempt_count, error_message))
        conn.commit()
        cursor.close()
        conn.close()
    
    except Exception as e:
        logger.error(f'Error in record_game_log_failure: {str(e)}')
        raise e
        
def record_boxscore_failure(game_id, attempt_count, error_message):
    """
    Stores failures while fethcning box scores in the boxscore_failure table in RDS.
    
    Paramters:
        - game_id (str): Unique identifier for the game_id for the failed boxscore.
        - attempt_count (int): The number of unsuccessful attepmts to retrieve the box scores.
        - error_message (str): A string containing the exception thrown while fetching box scores.
        
    Returns:
        - None
    """
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO boxscore_failures (game_id, attempt_count, error_message, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (game_id)
            DO UPDATE SET
              attempt_count = boxscore_failures.attempt_count + EXCLUDED.attempt_count,
              error_message = EXCLUDED.error_message,
              created_at = NOW();
        """
        cursor.execute(insert_query, (game_id, attempt_count, error_message))
        conn.commit()
        cursor.close()
        conn.close()
    
    except Exception as e:
        logger.error(f'Error in record_boxscore_failure: {str(e)}')
        raise e