import json
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
# Import utility functions from utils.py
from utils import (
    get_game_logs,
    process_game_logs,
    get_boxscores,
    process_boxscores
)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load the .env file
load_dotenv()

def lambda_handler(event, context):
    """
    AWS Lambda handler function for incremental data ingestion.
    This function:
    - Fetches yesterday's NBA game logs.
    - Processes game logs.
    - Fetches and processes boxscores for those games.
    
    Parameters:
    - event (dict): Event data that triggers the Lambda function.
    - context (object): Provides runtime information to the handler.
    
    Returns:
    - dict: Status message indicating success or failure.
    """
    
    try:
        # Set date for API call (yesterday)
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_str = yesterday.strftime('%m/%d/%Y')
        logger.info(f"Incremental load for NBA data game and box score data for {yesterday_str}")

        # Fetch game logs for yesterday
        game_logs_df = get_game_logs(yesterday_str, yesterday_str)
        if not game_logs_df.empty:
            # Process the retrieved game logs
            clean_game_logs_df = process_game_logs(game_logs_df)
            print(clean_game_logs_df.dtypes)
            logger.info(f"Retrieved and processed {len(clean_game_logs_df)} games.")

            # Fetch and process boxscores for each unique game
            boxscores_list = []
            unique_games = clean_game_logs_df['game_id'].unique()
            for game_id in unique_games:
                try:
                    boxscore_df = get_boxscores(game_id)
                    boxscores_list.append(boxscore_df)
                    logger.info(f"Box scores for game_id {game_id} retrieved successfully")
                except Exception as box_e:
                    logger.error(f"Failed to process box scores for game_id {game_id}: {str(box_e)}")
                    # Continue with the next game_id
                    continue

            if boxscores_list:
                boxscores_df = pd.concat(boxscores_list, ignore_index=True)
                clean_boxscores_df = process_boxscores(boxscores_df)
                print(clean_boxscores_df.dtypes)
                # TODO: store_boxscores_in_rds(clean_boxscores_df)
                # TODO: store_game_stats_in_rds(...) and store_games_in_rds(...) as needed
                logger.info(f"Processed boxscores for {len(boxscores_df)} records.")
            else:
                logger.info('No boxscores found for yesterday.')

        else:
            logger.info('No games found for yesterday.')

        return {
            'statusCode': 200,
            'body': json.dumps('Incremental data ingestion complete and stored successfully.')
        }
    
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Data ingestion failed: {str(e)}")
        }
    
# Entry Point (for local testing)
if __name__ == "__main__":
    lambda_handler({}, {})