import json
import boto3
import psycopg2
from nba_api.stats.endpoints import leaguegamefinder
import os

def lambda_handler(event, context):
    # Fetch data from NBA API
    gamefinder = leaguegamefinder.LeagueGameFinder()
    games = gamefinder.get_data_frames()[0]
    print(games)

# Test
event = {}
context = {} 
lambda_handler(event, context)