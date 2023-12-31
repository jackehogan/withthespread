from pymongo.mongo_client import MongoClient
import pandas as pd
import json
import datetime


def read_json(folder,file_name):
    with open(folder+'/'+file_name+'.txt') as i:
        return json.load(i)
def mongoConn():
    from pymongo.mongo_client import MongoClient
    config = read_json('data','config')
    USERNAME = config['mongo']['username']
    PW = config['mongo']['pw']

    uri = f"mongodb+srv://{USERNAME}:{PW}@cluster0.ml8jvfc.mongodb.net/?retryWrites=true&w=majority&appName=AtlasApp"
    # Create a new client and connect to the server
    client = MongoClient(uri)
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    return client

def get_db(client,mongo_db,mongo_collection):
    # mongo_db = 'withTheSpread'  # Replace with your desired database name
    # mongo_collection = 'bets'  # Replace with your desired collection name
    cursor = client[mongo_db][mongo_collection].find({})
    data = list(cursor)
    df = pd.DataFrame(data)

    return df

def add_to_db(client,mongo_db,mongo_collection,df):
    # mongo_db = 'withTheSpread'  # Replace with your desired database name
    # mongo_collection = 'season_spreads'  # Replace with your desired collection name
    # Create the database
    db = client[mongo_db]
    # Create the collection
    collection = db[mongo_collection]
    data = df.reset_index().rename(columns={'index': 'Team'}).to_dict(orient='records')
    collection.insert_many(data)

def getLastWeeksResults():
    import http.client
    import json
    config = read_json('data','config')
    conn = http.client.HTTPSConnection("v1.american-football.api-sports.io")
    API_KEY = config['results']['key']
    headers = {
        'x-rapidapi-host': "v1.american-football.api-sports.io",
        'x-rapidapi-key': API_KEY
    }

    conn.request("GET", "/games?league=1&season=2023", headers=headers)

    res = conn.getresponse()
    data = res.read()
    return data

def extractLastWeeksResults(data):
    games = [game for game in json.loads(data.decode("utf-8"))['response'] if
             game['game']['stage'] == 'Regular Season' and game['game']['status']['short'] != 'NS']
    weeks = sorted(list(set([game['game']['week'] for game in games])))
    weeks_num = sorted([int(week.split(' ')[1]) for week in weeks])
    lastweekstr = 'week' + str(max(weeks_num))
    nextweekstr = 'week' + str(max(weeks_num) + 1)
    print(lastweekstr, nextweekstr)
    print(weeks_num)
    gamescores = {}
    gamescorediffs = {}
    opponent = {}
    for game in games:
        if game['game']['week'] == weeks[-1]:
            gamescores[game['teams']['home']['name']] = game['scores']['home']['total']
            gamescores[game['teams']['away']['name']] = game['scores']['away']['total']
            gamescorediffs[game['teams']['home']['name']] = game['scores']['home']['total'] - game['scores']['away'][
                'total']
            gamescorediffs[game['teams']['away']['name']] = game['scores']['away']['total'] - game['scores']['home'][
                'total']
            # opponent[game['teams']['home']['name']] = game['teams']['away']['name']
            # opponent[game['teams']['away']['name']] = game['teams']['home']['name']

    dfgameresults = pd.concat([
        pd.Series(gamescores, name='score'),
        pd.Series(gamescorediffs, name='diff')
    ], axis=1)
    dfgameresults

    return dfgameresults, weeks_num,nextweekstr

def getNextWeeksSpreads():
    import requests
    config = read_json('data','config')
    API_KEY = config['spreads']['key']
    url = f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={API_KEY}'

    r = requests.get(url)

    r = requests.get(url)
    if r.status_code == 200:
        # for p in r.content:
        #     print(p)
        spreaddata = json.loads(r.content)

    return spreaddata

def extractNextWeeksSpreads(spreaddata, weeks_num):
    date_format = '%Y-%m-%dT%H:%M:%SZ'
    datetime.datetime.strptime(spreaddata[0]['commence_time'],
                               date_format) < datetime.datetime.today() + datetime.timedelta(days=7)
    nextweek = [game for game in spreaddata if datetime.datetime.strptime(game['commence_time'],
                                                                          date_format) < datetime.datetime.today() + datetime.timedelta(
        days=7)]
    games = [game['bookmakers'][0]['markets'] for game in
             nextweek]  # + [game['bookmakers'][1]['markets'] for game in thisweek]

    nextweek_spreads = {}
    order = {}
    opponent = {}
    for i, game in enumerate(games):
        for bet in game:
            if bet['key'] == 'spreads':
                nextweek_spreads[bet['outcomes'][0]['name']] = bet['outcomes'][0]['point']
                nextweek_spreads[bet['outcomes'][1]['name']] = bet['outcomes'][1]['point']
                order[bet['outcomes'][0]['name']] = i
                order[bet['outcomes'][1]['name']] = i
                opponent[bet['outcomes'][0]['name']] = bet['outcomes'][1]['name']
                opponent[bet['outcomes'][1]['name']] = bet['outcomes'][0]['name']
    nextweek_spreads
    dfseasonspreads_nextweek = pd.concat([pd.Series(nextweek_spreads, name='spread'),
                                          pd.Series(order, name='order'),
                                          pd.Series(opponent, name='opponent')
                                          ], axis=1)
    return dfseasonspreads_nextweek

def delete_documents(client, mongo_db, mongo_collection, field_to_match, value_to_match):
    """
    Delete documents from a MongoDB collection that match a certain value.

    Parameters
    ----------
    mongo_db : str
        Name of the MongoDB database.
    mongo_collection : str
        Name of the MongoDB collection.
    value_to_match : int or str
        Value to match for deletion.

    Returns
    -------
    None.
    """

    # Connect to the database and collection
    database = client[mongo_db]
    collection = database[mongo_collection]

    if field_to_match=='All' and value_to_match=='All':
        collection.delete_many({})
        print(f"Deleted all documents in the collection.")
        return


    # Create a query to find documents matching the value
    query = {field_to_match: value_to_match}  # Replace 'field_name_to_match' with your actual field name

    # Delete documents that match the query
    result = collection.delete_many(query)

    # Print the number of deleted documents
    print(f"Deleted {result.deleted_count} documents that matched the value.")

def update_document(client, mongo_db, mongo_collection, week, fields_to_update,df):
    """
    Update a document in a MongoDB collection.

    Parameters
    ----------
    client : pymongo.MongoClient
        A MongoClient instance connected to MongoDB.
    mongo_db : str
        Name of the MongoDB database.
    mongo_collection : str
        Name of the MongoDB collection.
    filter_query : dict
        A MongoDB filter query.
    update_query : dict
        A MongoDB update query.

    Returns
    -------
    None.
    """

    # Connect to the database and collection
    database = client[mongo_db]
    collection = database[mongo_collection]

    df_forUpdate = df[df['Week']==week]


    # Update the document
    for field in fields_to_update:
        for index, row in df_forUpdate.iterrows():
            filter_query = {"Week": row["Week"],'Team':row['Team']}  # Replace "field_name" with the actual identifier
            old_value = collection.find_one(filter_query)[field]
            update_query = {"$set": {field: row[field]}}  # Update other fields as needed
            result = collection.update_one(filter_query, update_query)
            new_value = collection.find_one(filter_query)[field]
            print(f"Updated week: {week}, team: {row['Team']} {field} from {old_value} to {new_value}.")
    # Print success message
    print("Updated document in the collection.")