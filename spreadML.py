import pandas as pd
import json
import datetime
import numpy as np
from xgboost import XGBRegressor,XGBClassifier
from sklearn.metrics import accuracy_score,f1_score,make_scorer
from sklearn.model_selection import cross_val_score

def genData(df,nextweek,nweeks):

    def genFrames(df,startweek,nweeks):
        ytemp = df[startweek+nweeks]
        ytemp = ytemp[ytemp!=0].dropna()
        Xtemp = df[[i for i in range(startweek,startweek+nweeks)]].loc[ytemp.index]
        Xtemp['Week'] = ytemp.name
        Xtemp = Xtemp.droplevel(2).reset_index().set_index(['Team','Year','Week']).rename_axis(None,axis=1)
        # Xtemp['Team'] = Xtemp['Team'].astype('category')
        # Xtemp['Year'] = Xtemp['Year'].astype('category')
        return Xtemp,ytemp#,year,week
    # dfspread = pd.read_csv('data/dfspread.csv',index_col=[0,1,2])
    dfspread = df[df['Bet']=='SpreadScore'].drop(['_id','Bet'],axis=1).set_index(['Team','Year','teamyearid'])
    dfspread.columns = [int(col) if col not in ['Team','Year','teamyearid'] else col for col in dfspread.columns]
    dfmodel = dfspread.loc[pd.IndexSlice[2022] != dfspread.index.get_level_values('Year')]
    dfval = dfspread.loc[pd.IndexSlice[2022] == dfspread.index.get_level_values('Year')]

    # thisweek = weeks_num[0]+1
    # nweeks = min(5,weeks_num[0])

    Xlist = [genFrames(dfmodel,startweek,nweeks)[0].reset_index() for startweek in range(1,nextweek+1-nweeks)]
    ylist = [genFrames(dfmodel,startweek,nweeks)[1] for startweek in range(1,nextweek+1-nweeks)]
    X = pd.DataFrame(np.concatenate(Xlist, axis=0))
    X.columns = ['Team','Year','Week']+list(range(1,nweeks+1))
    for col in X.columns:
        if col in ['Team','Year','Week']:
            X[col] = X[col].astype('category')
        else:
            X[col] = pd.to_numeric(X[col])
    y = pd.Series(np.concatenate(ylist, axis=0))

    from sklearn.model_selection import train_test_split
    import random
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBRegressor,XGBClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=random.randint(1,100)
                                                        # random_state=42
                                                        )

    Xlist = [genFrames(dfval,startweek,nweeks)[0].reset_index() for startweek in range(1,nextweek+1-nweeks)]
    ylist = [genFrames(dfval,startweek,nweeks)[1] for startweek in range(1,nextweek+1-nweeks)]
    X_val = pd.DataFrame(np.concatenate(Xlist, axis=0))
    X_val.columns = ['Team','Year','Week']+list(range(1,nweeks+1))
    for col in X_val.columns:
        if col in ['Team','Year','Week']:
            X_val[col] = X_val[col].astype('category')
        else:
            X_val[col] = pd.to_numeric(X_val[col])
    y_val = pd.Series(np.concatenate(ylist, axis=0))

    return X_train, X_test, y_train, y_test,X_val,y_val

# nextweek = weeks_num[0]+1
# nweeks = min(5,weeks_num[0])
# X_train, X_test, y_train, y_test,X_val,y_val = genData(nextweek,nweeks)

def fitxgbModel(modeltype,X_train, y_train):
    print('Model columns:', X_train.columns)
    from hyperopt import fmin, tpe, hp
    def objective(params):

        # selected_features = [i for i, use_feature in enumerate(params['features']) if use_feature]

        # if not any(params['features']):  # Ensure at least one feature is selected
        #     return 1.0  # Penalize if no features are selected

        params = {
            'learning_rate': params['learning_rate'],
            'max_depth': int(params['max_depth']),
            'n_estimators': int(params['n_estimators']),
            'reg_lambda': params['reg_lambda'],
            'reg_alpha': params['reg_alpha'],
            'min_child_weight': params['min_child_weight'],

            # 'scale_pos_weight': params['scale_pos_weight'],
            # Add more hyperparameters
        }
        modeldict = {'reg':XGBRegressor,'clas':XGBClassifier}
        model = modeldict[modeltype](**params,enable_categorical=True,tree_method="hist")
        from sklearn.preprocessing import LabelEncoder
        y_encoded = LabelEncoder().fit_transform(np.sign(y_train))
        # model = XGBRegressor(**params,enable_categorical=True,tree_method="hist")
        if modeltype=='clas':
            cvscore = -cross_val_score(model, X_train, y_encoded, cv=5, scoring='accuracy')
        else:
            cvscore = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

        return cvscore.mean()  # Hyperopt minimizes the objective, so we use negative accuracy

    # Define the search space for hyperparameters
    space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'max_depth': hp.quniform('max_depth', 2, 5, 1),
        'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
        'reg_lambda': hp.uniform('reg_lambda', 1, 5),
        'reg_alpha': hp.uniform('reg_alpha', 1, 5),
        'min_child_weight': hp.uniform('min_child_weight', 1, 5),
        # 'scale_pos_weight': hp.quniform('scale_pos_weight', 1, 10, 3),
        # 'features': [hp.choice(f'feature_{i}', [True, False]) for i in range(X[modelcolsall].shape[1])]

        # Add more hyperparameters to optimize
    }

    # Run hyperparameter optimization
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)
    print("Best hyperparameters:", best)

    def round_whole_numbers(d, decimal_places=2):
        rounded_dict = {}

        for key, value in d.items():
            if isinstance(value, (int, float)) and value.is_integer():
                rounded_value = int(value)
            else:
                rounded_value = value
            rounded_dict[key] = rounded_value

        return rounded_dict

    bestparams = round_whole_numbers(best)
    return bestparams