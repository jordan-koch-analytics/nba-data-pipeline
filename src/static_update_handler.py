import sys
import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from utils import (
    get_teams,
    get_players,
    # store_teams_in_rds,
    # store_players_in_rds
)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load the .env file
load_dotenv()

def lambda_handler(event, context):
    """
    AWS Lambda handler function to refresh static data on a weekly basis (teams and players).
    
    Parameters:
    - event (dict): Event data that triggers the Lambda function.
    - context (object): Provides runtime information to the handler.
    
    Returns:
    - dict: Status message indicating success or failure.
    """
    
    try:
        logger.info("Starting static data refresh...")

        # Fetch and process teams
        teams_df = get_teams()
        print(teams_df.dtypes)
        if not teams_df.empty:
            # store_teams_in_rds(teams_df)
            logger.info(f"Updated teams in RDS with {len(teams_df)} records.")
        else:
            logger.warning("No teams data fetched.")

        # Fetch and process players
        players_df = get_players()
        print(players_df.dtypes)
        if not players_df.empty:
            # store_players_in_rds(players_df)
            logger.info(f"Updated players in RDS with {len(players_df)} records.")
        else:
            logger.warning("No players data fetched.")

        logger.info("Static data refresh completed successfully.")

        return {
            'statusCode': 200,
            'body': json.dumps('Static data refresh complete.')
        }

    except Exception as e:
        logger.error(f"Error in static_update_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Static data refresh failed: {str(e)}")
        }

# Entry Point (for local testing)
if __name__ == "__main__":
    lambda_handler({}, {})