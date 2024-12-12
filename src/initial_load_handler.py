import json
import logging
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from utils import (
    get_teams,
    get_players,
    process_teams,
    process_players,
    get_game_logs,
    process_game_logs,
    get_boxscores,
    process_boxscores,
    # store_teams_in_rds,
    # store_players_in_rds,
    # store_games_in_rds,
    # store_boxscores_in_rds
)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load the .env file
load_dotenv()

def lambda_handler(event, context):
    """
    AWS Lambda handler function for the initial data ingestion.
    This function:
    - Fetches and stores static data (teams and players).
    - Fetches game logs from the season start date up to yesterday.
    - Fetches and stores boxscores for all games retrieved.
    
    Parameters:
    - event (dict): Event data that triggers the Lambda function.
    - context (object): Provides runtime information to the handler.
    
    Returns:
    - dict: Status message indicating success or failure.
    """
    try:
        logger.info("Starting initial data load...")

        # Fetch and store static data (teams and players)
        teams_df = get_teams()
        if not teams_df.empty:
            # store_teams_in_rds(teams_df)
            logger.info(f"Stored {len(teams_df)} teams in RDS successfully.")
        else:
            logger.warning("No teams data fetched.")
        print('Teams Done.')

        players_df = get_players()
        if not players_df.empty:
            # store_players_in_rds(players_df)
            logger.info(f"Stored {len(players_df)} players in RDS successfully.")
        else:
            logger.warning("No players data fetched.")
        print('Players Done.')

        # Set date range for initial load
        season_start_date = datetime(2024, 12, 5)
        start_date_str = season_start_date.strftime('%m/%d/%Y')
        end_date = datetime.now() - pd.Timedelta(days=1)
        end_date_str = end_date.strftime('%m/%d/%Y')

        # Fetch game logs from date range
        logger.info(f"Fetching game logs from {start_date_str} to {end_date_str}")

        game_logs_df = get_game_logs(start_date_str, end_date_str)
        if not game_logs_df.empty:
            # Process the retrieved game logs
            clean_game_logs_df = process_game_logs(game_logs_df)
            print(clean_game_logs_df)
            logger.info(f"Retrieved and processed {len(clean_game_logs_df)} games.")

            # Store game logs in RDS
            # store_games_in_rds(game_logs_df)
            logger.info("Game logs data successfully stored in RDS.")

            # Fetch, process, and store boxscores for each unique game
            boxscores_list = []
            unique_games = clean_game_logs_df['game_id'].unique()
            for game_id in unique_games:
                try:
                    boxscore_df = get_boxscores(game_id)
                    if not boxscore_df.empty:
                        boxscores_list.append(boxscore_df)
                        print(f'GAME_ID: {game_id}')
                        print(boxscore_df)
                        print()
                        logger.info(f"Box scores for game_id {game_id} retrieved successfully.")
                except Exception as box_e:
                    logger.error(f"Failed to process box scores for game_id {game_id}: {str(box_e)}")
                    # TODO Store failure and retry 
                    continue

            if boxscores_list:
                boxscores_df = pd.concat(boxscores_list, ignore_index=True)
                clean_boxscores_df = process_boxscores(boxscores_df)
                print(clean_boxscores_df)
                # store_boxscores_in_rds(clean_boxscores_df)
                logger.info(f"Stored {len(clean_boxscores_df)} boxscores in RDS successfully.")
            else:
                logger.info("No boxscores found to store.")

        else:
            logger.info("No games found for the specified date range.")

        logger.info("Initial data load completed successfully.")

        return {
            'statusCode': 200,
            'body': json.dumps('Initial data ingestion complete and stored successfully.')
        }

    except Exception as e:
        logger.error(f"Error in initial_load_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Initial data ingestion failed: {str(e)}")
        }

# Entry Point (for local testing)
if __name__ == "__main__":
    lambda_handler({}, {})