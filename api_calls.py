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

def getGameResults(year):
    import http.client
    import json
    config = read_json('data','config')
    conn = http.client.HTTPSConnection("v1.american-football.api-sports.io")
    API_KEY = config['results']['key']
    headers = {
        'x-rapidapi-host': "v1.american-football.api-sports.io",
        'x-rapidapi-key': API_KEY
    }

    conn.request("GET", f"/games?league=1&season={year}", headers=headers)

    res = conn.getresponse()
    data = res.read()
    return data

def extractLastWeeksResults(data,week=None):
    games = [game for game in json.loads(data.decode("utf-8"))['response'] if
             game['game']['stage'] == 'Regular Season' and game['game']['status']['short'] != 'NS']
    weeks = sorted(list(set([game['game']['week'] for game in games])))
    int_weeks = sorted([int(week.split(' ')[1]) for week in weeks])
    if not week: week = int_weeks[-1]+1
    int_weeks = list(set(int_weeks+[week]))
        # if late in running routine
    # if late in running routine
    # int_weeks = int_weeks[:-1]

    if week:
        week = round(week)
        lastweekstr = 'Week ' + str(week-1)
        nextweekstr = 'Week ' + str(week)
        int_weeks = [w for w in int_weeks if w <= week]
    else:
        lastweekstr = 'Week ' + str(int_weeks[-1]-1)
        nextweekstr = 'Week ' + str(int_weeks[-1])
        # int_weeks = [w for w in int_weeks if w < max(int_weeks)]
        # week = max(int_weeks)
    print(lastweekstr, nextweekstr)
    print(int_weeks)
    print(f'Last week is {lastweekstr} and next week is {nextweekstr}')
    gamescores = {}
    gamescorediffs = {}
    opponent = {}
    for game in games:
        # if True:
        # if game['game']['week'] == 'Week '+str(max(int_weeks)):
        if game['game']['week'] == lastweekstr:
            gamescores[game['teams']['home']['name']] = game['scores']['home']['total']
            gamescores[game['teams']['away']['name']] = game['scores']['away']['total']
            gamescorediffs[game['teams']['home']['name']] = game['scores']['home']['total'] - game['scores']['away'][
                'total']
            gamescorediffs[game['teams']['away']['name']] = game['scores']['away']['total'] - game['scores']['home'][
                'total']
            # opponent[game['teams']['home']['name']] = game['teams']['away']['name']
            # opponent[game['teams']['away']['name']] = game['teams']['home']['name']
    datesUTC = [game['game']['date']['date']+'T'+game['game']['date']['time']+':00Z' for game in games if game['game']['week'] == nextweekstr]
    # datesUTC = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in datesUTC]
    # get unique dates
    datesUTC = list(set(datesUTC))
    # datesUTC = []
    print(datesUTC)

    dfgameresults = pd.concat([
        pd.Series(gamescores, name='score'),
        pd.Series(gamescorediffs, name='diff')
    ], axis=1)
    dfgameresults
    breakpoint()
    return dfgameresults, int_weeks, nextweekstr, datesUTC

def getNextWeeksSpreads(dates=None, key_type='free'):
    import requests
    config = read_json('data','config')
    API_KEY = config['spreads'][f'key_{key_type}']
    SPORT = 'americanfootball_nfl'
    REGIONS = 'us'
    MARKETS = 'h2h,spreads,totals'
    ODDS_FORMAT = 'american'
    # get max date in dates and convert to this format ''2023-10-10T12:15:00Z'

    if key_type=='free':
        DATE = datetime.datetime.today().strftime('%Y-%m-%d')
    else:
        DATE = min(dates)
        datetime_format = '%Y-%m-%dT%H:%M:%SZ'
        minDate = datetime.datetime.strptime(min(dates), datetime_format).date()
        maxDate = datetime.datetime.strptime(max(dates), datetime_format).date()

    if key_type == 'free': url = f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={API_KEY}'
    else: url = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}&oddsFormat={ODDS_FORMAT}'
    if key_type == 'paid': url = url + f'&date={DATE}'

    r = requests.get(url)
    if r.status_code == 200:
        # for p in r.content:
        #     print(p)
        spreaddata = json.loads(r.content)
    breakpoint()
    print('check spreaddata')
    if key_type == 'paid': spreaddata = spreaddata['data']
    #     hist_spreaddata = json.loads(r.content)
    # spreaddata = hist_spreaddata['data']

    print(f'number of games in spreaddata before date filter = {len(spreaddata)}')
    # filter out games that are not in the date range
    # spreaddata = [game for game in spreaddata if game['commence_time'] >= minDate and game['commence_time'] <= maxDate]
    if key_type == 'paid':
        spreaddata = [game for game in spreaddata if datetime.datetime.strptime(game['commence_time'],datetime_format).date() >= minDate and datetime.datetime.strptime(game['commence_time'],datetime_format).date() <= maxDate]
        print(f'number of games in spreaddata after date filter = {len(spreaddata)}')
    print('Remaining requests', r.headers['x-requests-remaining'])
    print('Used requests', r.headers['x-requests-used'])
    breakpoint()
    return spreaddata

def extractNextWeeksSpreads(spreaddata,datesUTC):
    datetime_format = '%Y-%m-%dT%H:%M:%SZ'
    date_format = '%Y-%m-%d'
    nextweek = spreaddata
    nextweek = [game for game in spreaddata if datetime.datetime.strptime(game['commence_time'],datetime_format)
                < datetime.datetime.today() + datetime.timedelta(days=7)]
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
    breakpoint()
    # check next week spreads
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
        print(f"Deleted all documents {mongo_collection} in the collection.")
        return


    # Create a query to find documents matching the value
    query = {field_to_match: value_to_match}  # Replace 'field_name_to_match' with your actual field name

    # Delete documents that match the query
    result = collection.delete_many(query)

    # Print the number of deleted documents
    print(f"Deleted {result.deleted_count} documents that matched the value.")

def update_document(client, mongo_db, mongo_collection, week, year, fields_to_update,df):
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

    df_forUpdate = df[(df['Week']==week) & (df['Year']==year)]
    breakpoint()


    # Update the document
    for field in fields_to_update:
        for index, row in df_forUpdate.iterrows():
            filter_query = {"Week": row["Week"],"Year":row["Year"] ,'Team':row['Team']}  # Replace "field_name" with the actual identifier
            old_value = collection.find_one(filter_query)[field]
            update_query = {"$set": {field: row[field]}}  # Update other fields as needed
            result = collection.update_one(filter_query, update_query)
            new_value = collection.find_one(filter_query)[field]
            print(f"Updated week: {week}, year: {year}, team: {row['Team']} {field} from {old_value} to {new_value}.")
    # Print success message
    print("Updated document in the collection.")