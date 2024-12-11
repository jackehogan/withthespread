import pandas as pd
import numpy as np
import datetime
from api_calls import mongoConn,get_db,add_to_db,getGameResults,extractLastWeeksResults,extractNextWeeksSpreads,getNextWeeksSpreads,update_document,delete_documents
from spreadML import genData,fitxgbModel
from xgboost import XGBRegressor,XGBClassifier

### Get data from mongoDB
client = mongoConn()
dfbets = get_db(client,'withTheSpread','bets')
dfseasonspreads = get_db(client,'withTheSpread','season_spreads')
dfseasonspreads = dfseasonspreads.set_index(['Team'],drop=True).drop(['_id'],axis=1)

### Get last week's results
data = getGameResults()
dfgameresults,weeks_num,nextweekstr = extractLastWeeksResults(data,1)
breakpoint()

dfseasonspreads_lastweek = dfseasonspreads[dfseasonspreads['Week']==weeks_num[-1]].drop(['diff','score'],axis=1).merge(dfgameresults,left_index=True,right_index=True,how='left')
dfseasonspreads_lastweek['spreadscore'] = dfseasonspreads_lastweek['diff'] + dfseasonspreads_lastweek['spread']
print(dfseasonspreads_lastweek)
#print San Francisco 49ers result
print('Checking niners last weeks results: ',dfseasonspreads_lastweek.loc['San Francisco 49ers'])
breakpoint()

### update seasonspreads with last week's results
#### must check this step!!!!!!!!!!!!!
dfseasonspreads.loc[dfseasonspreads['Week']==weeks_num[-1],'score'] = dfseasonspreads_lastweek['score']
dfseasonspreads.loc[dfseasonspreads['Week']==weeks_num[-1],'diff'] = dfseasonspreads_lastweek['diff']
dfseasonspreads.loc[dfseasonspreads['Week']==weeks_num[-1],'spreadscore'] = dfseasonspreads_lastweek['spreadscore']
#alternatively could use this below
# dfseasonspreads.loc[dfseasonspreads['Week']==weeks_num[-1],['score','diff','spreadscore']].update(dfseasonspreads_lastweek[['score','diff','spreadscore']]
print('Niners season totals:' , dfseasonspreads.loc['San Francisco 49ers'])
### get next week's spreads
spreaddata = getNextWeeksSpreads()
dfseasonspreads_nextweek = extractNextWeeksSpreads(spreaddata,weeks_num)
print(f'Next weeks 49ers spread is {dfseasonspreads_nextweek.loc["San Francisco 49ers"]}')
breakpoint()
### train model

#import xgb models and metrics
from xgboost import XGBRegressor,XGBClassifier
## create training data
nextweek = weeks_num[-1]+1
nweeks = min(5,weeks_num[-1])
print(f'Using {nweeks} weeks of data to predict next week {nextweek}')
X_train, X_test, y_train, y_test,X_val,y_val = genData(dfbets,nextweek,nweeks)

## fit best parameters for  reg and clas models
bestparamsreg = fitxgbModel('reg',X_train,y_train,X_test,y_test)
bestparamsclas = fitxgbModel('clas',X_train,y_train,X_test,y_test)

## train xgb models
reg = XGBRegressor(**bestparamsreg,enable_categorical=True,tree_method="hist")
reg.fit(X_train,y_train)
clas = XGBClassifier(**bestparamsclas,enable_categorical=True,tree_method="hist")
from sklearn.preprocessing import LabelEncoder
y_encoded = LabelEncoder().fit_transform(np.sign(y_train))
clas.fit(X_train,y_encoded)

## predict next week's spreads
X = dfseasonspreads.pivot(columns='Week',values='spreadscore')
X = X.iloc[:,-nweeks:] ##only take last nweeks
X.columns = [f'{n}_weeksago' for n in range(nweeks,0,-1)]
X = X.reset_index().rename({'index':'Team'},axis=1)
X['Year'] = 2024
X['Week'] = weeks_num[-1]+1
X# .columns[-1].split(' ')[1])-1],'Team','Year','Week']
X = X[['Team','Year','Week']+[col for col in X.columns if col not in ['Team','Year','Week']]]
for col in ['Team','Year','Week']:
    X[col] = X[col].astype('category')
if weeks_num[-1]+1 in X.columns:
    X = X.drop([weeks_num[-1]+1],axis=1)
# X = X.set_index(['Team','Year'])
X
## for reasons not well understood by the writer this prediction is extradinarily bad, consistently predicting the wrong outcome. Knowing this the predictions will be flipped - effective in week 11 2023
dfseasonspreads_temp = pd.concat([X.set_index('Team'),
           # pd.DataFrame(clas.predict_proba(X),index=X['Team']).iloc[:,1].rename('coverprob'),  ## this is the correct prediction
           # pd.Series(reg.predict(X),index=X['Team'],name='predspread')],axis=1)   ## this is the correct prediction, flipping to see if predictions improve. Implications are not good for our career as a data scientist
            pd.DataFrame(1-clas.predict_proba(X), index=X['Team']).iloc[:, 1].rename('coverprob'),
            pd.Series(-reg.predict(X), index=X['Team'], name='predspread')], axis = 1)
dfseasonspreads_temp = dfseasonspreads_temp[['Year','Week','coverprob','predspread']].join(dfseasonspreads_nextweek)
dfseasonspreads_temp['coverprob_diff'] = dfseasonspreads_temp.apply(lambda row: np.nan if pd.isnull(row['opponent']) else row['coverprob'] - dfseasonspreads_temp.loc[row['opponent']]['coverprob'],axis=1)
dfseasonspreads_temp['predspread_diff'] = dfseasonspreads_temp.apply(lambda row: np.nan if pd.isnull(row['opponent']) else row['predspread'] - dfseasonspreads_temp.loc[row['opponent']]['predspread'],axis=1)

breakpoint()
#### happy with spread routine outcome? Next steps will save to mongoDB and csv
print(dfseasonspreads_temp)
dfseasonspreads_full = pd.concat([dfseasonspreads,dfseasonspreads_temp])
breakpoint()
#
# dfseasonspreads_full.to_csv('data/Season_spreads.csv')
# dfseasonspreads_temp.to_csv(f'data/Season_spreads {nextweekstr}.csv')
#
# breakpoint()
# ## check to make sure following steps update last weeks data correctly
# dfupdate = dfseasonspreads_full[dfseasonspreads_full['Week']==weeks_num[-1]]
# ##update last week's data in mongoDB (score,diff,spreadscore)
# update_document(client,'withTheSpread','season_spreads',weeks_num[-1],['score','diff','spreadscore'],dfupdate.reset_index())
# ## update next weeks data, all columns, score,diff,spreadscore are null
# add_to_db(client,'withTheSpread','season_spreads',dfseasonspreads_full[dfseasonspreads_full['Week']==nextweek])
# ##replace backup db with full season df
# delete_documents(client, 'withTheSpread', 'season_spreads_backup', 'All', 'All')
# add_to_db(client,'withTheSpread','season_spreads_backup',dfseasonspreads_full)
# client.close()