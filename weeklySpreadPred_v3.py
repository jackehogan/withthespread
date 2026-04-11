import pandas as pd
import numpy as np
import datetime
from api_calls import mongoConn,get_db,add_to_db,extractLastWeeksResults,extractNextWeeksSpreads,getNextWeeksSpreads,update_document,delete_documents
from spreadML import genData,fitxgbModel
from xgboost import XGBRegressor,XGBClassifier
from espn_nfl_scrape import get_nfl_scores_bs, nfl_week_sunday
# from weeklySpreadPred import dfseasonspreads_nextweek


class wts_nfl:
    def __init__(self,year=None,week=None, api_pay_type='paid'):
        self.year = year
        if week:
            self.week = week
        else:
            todays_date = datetime.datetime.now().date()
            nfl_sundays = nfl_week_sunday(year)
            # check which index is larger than today's date
            week = int(np.argmax([sunday > todays_date for sunday in nfl_sundays]))+ 1
            print(f'No week specified, using week {week} of {year}')
            self.week = week
        self.client = mongoConn()
        self.api_pay_type = api_pay_type

        self.dfbets = get_db(self.client, 'withTheSpread', 'bets')
        self.dfseasonspreads = get_db(self.client, 'withTheSpread', 'season_spreads')



    def loadDataset(self, week=None,year=None):
        # dfbets = get_db(client, 'withTheSpread', 'bets')
        # self.dfseasonspreads = get_db(client, 'withTheSpread', 'season_spreads')
        week = self.week if week is None else week
        year = self.year if year is None else year
        dfseasonspreads = self.dfseasonspreads
        dfseasonspreads = dfseasonspreads.set_index(['Team'], drop=True).drop(['_id'], axis=1)
        breakpoint()
        nfl_sunday_date = nfl_week_sunday(self.year,self.week)
        week_start = nfl_sunday_date - datetime.timedelta(days=3)
        week_end = nfl_sunday_date + datetime.timedelta(days=3)
        if week != 1:
            df_scrape = get_nfl_scores_bs(year,week-1)
            df_scrape['diff'] = df_scrape['home_score'] - df_scrape['away_score']
            df_scrape_home = df_scrape[['home_team', 'home_score','diff','away_team']].rename(columns={'home_team': 'Team', 'home_score': 'score', 'away_team':'opponent'}).set_index('Team')
            df_scrape_away = df_scrape[['away_team', 'away_score','diff','home_team']].rename(columns={'away_team': 'Team', 'away_score': 'score', 'home_team':'opponent'}).set_index('Team')
            df_scrape_away['diff'] = -df_scrape_away['diff']
            df_scrape_all = pd.concat([df_scrape_home, df_scrape_away]).sort_index()
            mask = (dfseasonspreads['Week'] == week-1) & (dfseasonspreads['Year'] == year)
            # df_scrape_all is indexed by team and has columns ['score','diff']
            vals = df_scrape_all[['score', 'diff']].reindex(dfseasonspreads.loc[mask].index)
            dfseasonspreads.loc[mask, ['score', 'diff']] = vals.values
            dfseasonspreads['spreadscore'] = dfseasonspreads['diff'] + dfseasonspreads['spread']

        # breakpoint()
        # if year is None, get the current year
        # if year is None:
        #     year = datetime.datetime.now().year
        # print(f'Year is {year}')
        # ### Get last week's results
        # data = getGameResults(year)
        # dfgameresults, int_weeks, nextweekstr, datesUTC = extractLastWeeksResults(data, week=week)


        # dfseasonspreads_lastweek = dfseasonspreads[(dfseasonspreads['Week'] == int_weeks[-1]-1) & (dfseasonspreads['Year'] == year)].drop(['diff', 'score'],
        #                                                                                                                                   axis=1).merge(dfgameresults, left_index=True, right_index=True, how='left')
        # dfseasonspreads_lastweek['spreadscore'] = dfseasonspreads_lastweek['diff'] + dfseasonspreads_lastweek['spread']
        # print(dfseasonspreads_lastweek)
        # # print San Francisco 49ers result
        # if int_weeks[-1] != 1: print('Checking niners last weeks results: ', dfseasonspreads_lastweek.loc['San Francisco 49ers'])

        ### update seasonspreads with last week's results
        #### must check this step!!!!!!!!!!!!!
        # breakpoint()
        # if int_weeks[-1] != 1:
            # dfseasonspreads.loc[dfseasonspreads['Week'] == int_weeks[-2], 'score'] = dfseasonspreads_lastweek['score']
            # dfseasonspreads.loc[dfseasonspreads['Week'] == int_weeks[-2], 'diff'] = dfseasonspreads_lastweek['diff']
            # dfseasonspreads.loc[dfseasonspreads['Week'] == int_weeks[-2], 'spreadscore'] = dfseasonspreads_lastweek['spreadscore']
            # dfseasonspreads.loc[(dfseasonspreads['Week'] == int_weeks[-1]-1) & (dfseasonspreads['Year'] == year), ['score', 'diff', 'spreadscore']] = dfseasonspreads_lastweek[['score', 'diff', 'spreadscore']]
            # alternatively could use this below
            # Check which rows will be updated
            # Define the condition for filtering `dfseasonspreads`
            # condition = (dfseasonspreads['Week'] == int_weeks[-2]) & (dfseasonspreads['Year'] == year)
            #
            # # Identify the indices to update in `dfseasonspreads`
            # indices_to_update = dfseasonspreads[condition].index
            #
            # # Reindex `dfseasonspreads_lastweek` to ensure it matches `indices_to_update` order,
            # # then perform the update
            # dfseasonspreads.loc[condition, ['score', 'diff', 'spreadscore']] = (dfseasonspreads_lastweek[['score', 'diff', 'spreadscore']].reindex(indices_to_update)
            # )
            # dfseasonspreads.loc[(dfseasonspreads['Week']==int_weeks[-2]) & (dfseasonspreads['Year']==year),['score','diff','spreadscore']].update(dfseasonspreads_lastweek[['score','diff','spreadscore']])
            print('Niners season totals:', dfseasonspreads.loc['San Francisco 49ers'])
            breakpoint()
        ### get next week's spreads
        if week == 19:
            dfseasonspreads_nextweek = None
        else:
            datesUTC = [week_start,week_end]
            spreaddata = getNextWeeksSpreads(datesUTC,api_pay_type)
            dfseasonspreads_nextweek = extractNextWeeksSpreads(spreaddata, datesUTC)
            if "San Francisco 49ers" not in dfseasonspreads_nextweek.index:
                print('No 49ers data found for next week, check to make sure they have a bye week')
                team_idx0 = dfseasonspreads_nextweek.index[0]
                print(f'Next weeks {team_idx0} spread is {dfseasonspreads_nextweek.loc[team_idx0]}')
            else:
                print(f'Next weeks 49ers spread is {dfseasonspreads_nextweek.loc["San Francisco 49ers"]}')
        self.dfseasonspreads = dfseasonspreads
        self.dfseasonspreads_nextweek = dfseasonspreads_nextweek
        return dfseasonspreads, dfseasonspreads_nextweek

    def predictSpreadscore(self,lookback_weeks):
        # import xgb models and metrics
        ## create training data
        week = self.week
        nweeks = min(lookback_weeks, self.week-1)
        dfseasonspreads = self.dfseasonspreads.copy()
        dfseasonspreads_nextweek = self.dfseasonspreads_nextweek
        if week == 1:
            print('in week 1 conditional')
            dfseasonspreads_temp = self.dfseasonspreads_nextweek.copy()
            dfseasonspreads_temp[['Year','Week','coverprob','predspread','coverprob_diff', 'predspread_diff']] = [year,week,.5,0,0,0]
        else:
            print(f'Using {nweeks} weeks of data to predict next week {week}')
            ''

            X_train, X_test, y_train, y_test, X_val, y_val = genData(self.dfbets, week, nweeks)

            ## fit best parameters for  reg and clas models
            y_train_enc = np.sign(y_train).replace({-1: 0, 1: 1})
            bestparamsreg = fitxgbModel('reg', X_train, y_train, y_train_enc)
            bestparamsclas = fitxgbModel('clas', X_train, y_train, y_train_enc)

            ## train xgb models
            reg = XGBRegressor(**bestparamsreg, enable_categorical=True, tree_method="hist")
            reg.fit(X_train, y_train)
            print(f'regression train score {reg.score(X_train, y_train)}')
            print(f'regression test score {reg.score(X_test, y_test)}')
            print(f'regression val score {reg.score(X_val, y_val)}')
            clas = XGBClassifier(**bestparamsclas, enable_categorical=True, tree_method="hist")
            # from sklearn.preprocessing import LabelEncoder
            # label_encoder = LabelEncoder()
            # y_encoded = LabelEncoder().fit_transform(np.sign(y_train))
            y_train_enc = np.sign(y_train).replace({-1: 0, 1: 1})
            clas.fit(X_train, y_train_enc)
            print(f'classifier train score {clas.score(X_train, y_train_enc)}')
            y_test_enc = np.sign(y_test).replace({-1: 0, 1: 1})
            print(f'classifier test score {clas.score(X_test, y_test_enc)}')
            y_val_enc = np.sign(y_val).replace({-1: 0, 1: 1})
            print(f'classifier val score {clas.score(X_val, y_val_enc)}')


            ## predict next week's spreads
            X = dfseasonspreads[(dfseasonspreads['Year']==year) & (dfseasonspreads['Week']<week)].pivot(columns='Week', values='spreadscore')
            # X = X.drop(columns=int_weeks[-1], axis=1)
            X = X.iloc[:, -nweeks:]  ##only take last nweeks
            X.columns = [f'{n}_weeksago' for n in range(nweeks, 0, -1)]
            X = X.reset_index().rename({'index': 'Team'}, axis=1)
            if week == 1: X =  dfseasonspreads.index.to_frame().drop_duplicates()
            X['Year'] = year
            X['Week'] = week
            X  # .columns[-1].split(' ')[1])-1],'Team','Year','Week']
            # X = X[['Team', 'Year', 'Week'] + [col for col in X.columns if col not in ['Team', 'Year', 'Week']]]
            X.drop(['Team', 'Year', 'Week'], axis=1) # gpt mods
            for col in ['Team', 'Year', 'Week']:
                print('Converting', col, 'to category')
                X[col] = X[col].astype('category')
            # if int_weeks[-1] in X.columns:
            #     print('Dropping week', int_weeks[-1])
            #     X = X.drop([int_weeks[-1]], axis=1)
            # X = X.set_index(['Team','Year'])

            ## for reasons not well understood by the writer this prediction is extradinarily bad, consistently predicting the wrong outcome. Knowing this the predictions will be flipped - effective in week 11 2023
            dfseasonspreads_temp = pd.concat([X.set_index('Team'),
                                              # pd.DataFrame(clas.predict_proba(X),index=X['Team']).iloc[:,1].rename('coverprob'),  ## this is the correct prediction
                                              # pd.Series(reg.predict(X),index=X['Team'],name='predspread')],axis=1)   ## this is the correct prediction, flipping to see if predictions improve. Implications are not good for our career as a data scientist
                                              pd.DataFrame(1 - clas.predict_proba(X[clas.get_booster().feature_names]), index=X['Team']).iloc[:, 1].rename('coverprob'),
                                              pd.Series(-reg.predict(X[reg.get_booster().feature_names]), index=X['Team'], name='predspread')], axis=1)
            dfseasonspreads_temp = dfseasonspreads_temp[['Year', 'Week', 'coverprob', 'predspread']].join(
                dfseasonspreads_nextweek)
            breakpoint()
            dfseasonspreads_temp['coverprob_diff'] = dfseasonspreads_temp.apply(
                lambda row: np.nan if pd.isnull(row['opponent']) else row['coverprob'] -
                                                                      dfseasonspreads_temp.loc[row['opponent']]['coverprob'],
                axis=1)
            dfseasonspreads_temp['predspread_diff'] = dfseasonspreads_temp.apply(
                lambda row: np.nan if pd.isnull(row['opponent']) else row['predspread'] -
                                                                      dfseasonspreads_temp.loc[row['opponent']]['predspread'],
                axis=1)

        self.dfseasonspreads = dfseasonspreads
        self.dfseasonspreads_temp = dfseasonspreads_temp
        return dfseasonspreads, dfseasonspreads_temp

    def updateDatabase(self):
        week = self.week
        year = self.year
        #### happy with spread routine outcome? Next steps will save to mongoDB and csv
        if week == 19:
            dfseasonspreads_full = self.dfseasonspreads
        else:
            dfseasonspreads_full = pd.concat([self.dfseasonspreads, self.dfseasonspreads_temp])
        # print 49ers last week results
        if week-1 in dfseasonspreads_full['Week'].unique(): 
            try:
                print('Niners last week update results totals:',  dfseasonspreads_full[(dfseasonspreads_full['Week'] == week-1) & (dfseasonspreads_full['Year'] == year)].loc['San Francisco 49ers'])
            except:
                print('no niners oops')
        # if week != 1: print('Niners last week update results totals:',  dfseasonspreads_full[(dfseasonspreads_full['Week'] == int_weeks[-2]) & (dfseasonspreads_full['Year'] == year)].loc['San Francisco 49ers'])
        breakpoint()
        if week != 19:
            try:
                print('Niners next week predictions totals:',  dfseasonspreads_full[(dfseasonspreads_full['Week'] == week) & (dfseasonspreads_full['Year'] == year)].loc['San Francisco 49ers'])
            except:
                print('No niners next week predictions')
        breakpoint()


        dfseasonspreads_full.to_csv('data/Season_spreads.csv')
        if week != 19:
            self.dfseasonspreads_temp.to_csv(f'data/Season_spreads week{week}-{year}.csv')

        breakpoint()
        print('update database is next. Check dfupdate')
        if week != 1:
            ## check to make sure following steps update last weeks data correctly
            dfupdate = dfseasonspreads_full[(dfseasonspreads_full['Week'] == week-1) & (dfseasonspreads_full['Year'] == year)]
            ##update last week's data in mongoDB (score,diff,spreadscore)
            update_document(self.client,'withTheSpread','season_spreads',week, year, ['score','diff','spreadscore'],dfupdate.reset_index())
        ## update next weeks data, all columns, score,diff,spreadscore are null
        if week != 19:
            breakpoint()
            self.dfseasonspreads_temp
            add_to_db(self.client
                      ,'withTheSpread'
                      ,'season_spreads'
                      ,self.dfseasonspreads_temp
                      # ,dfseasonspreads_full[(dfseasonspreads_full['Week']==week) & (dfseasonspreads_full['Year']==year)]
                      )
        ##replace backup db with full season df
        delete_documents(self.client, 'withTheSpread', 'season_spreads_backup', {"All": "All"})
        add_to_db(self.client,'withTheSpread','season_spreads_backup',dfseasonspreads_full)

if __name__ == '__main__':
    ### Get data from mongoDB
    # client = mongoConn()
    year = 2025
    # week = 15
    api_pay_type = 'paid'
    wts_nfl_pred = wts_nfl(year=year
                           # , week=week
                           , api_pay_type=api_pay_type)
    wts_nfl_pred.loadDataset()
    wts_nfl_pred.predictSpreadscore(lookback_weeks=5)
    wts_nfl_pred.updateDatabase()
    # if not week: week = int_weeks[-1]
    ### train model
    # if week != 19:
    #     dfseasonspreads, dfseasonspreads_temp = predictSpreadscore(lookback_weeks=5)
    # updateDatabase()
    # client.close()