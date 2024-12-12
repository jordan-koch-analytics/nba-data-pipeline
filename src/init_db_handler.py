import json
import logging
import os
from dotenv import load_dotenv
from utils import get_db_connection, read_sql_file

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load the .env file
load_dotenv()

def lambda_handler(event, context):
    """
    AWS Lambda handler to initialize the database schema and create tables.
    
    Parameters:
        - event (dict): Event data that triggers the Lambda function.
        - context (object): Provides runtime information to the handler.
    
    Returns:
        - dict: Status message indicating success or failure.
    """
    
    try:
        logger.info("Starting database initialization...")

        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Path to the schema SQL files
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd() # local testing
        schema_dir = os.path.normpath(os.path.join(base_dir, '..', 'sql', 'schema'))

        # List of schema files
        schema_files = [
            "create_teams_table.sql",
            "create_players_table.sql",
            "create_game_logs_table.sql",
            "create_boxscores_table.sql"
        ]

        for file_name in schema_files:
            file_path = os.path.join(schema_dir, file_name)
            sql = read_sql_file(file_path)
            cursor.execute(sql)
            logger.info(f"Executed schema file: {file_name}")

        conn.commit()
        cursor.close()
        conn.close()

        logger.info("Database schema initialized successfully.")
        return {
            'statusCode': 200,
            'body': json.dumps('Database initialization complete.')
        }

    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Initialization failed: {str(e)}")
        }
    
    
# Entry Point (for local testing)
if __name__ == "__main__":
    lambda_handler({}, {})