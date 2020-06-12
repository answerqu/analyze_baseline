import sklearn.metrics as skmetrics
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
import numpy as np
from sklearn.model_selection import cross_val_score

def hp_model(model,X,y,evals=100,max_iterations=500,metric=skmetrics.f1_score,dict_concat={}):
    def hyperopt_xgb_score(params):
        clf = XGBClassifier(**params)
        # усреднение по 3ем фолдам, для уменьшения влияния стахостичности
        # для ускорения можно использовать train_test_split один раз
        current_score = cross_val_score(clf, X, y,scoring=metric).mean()
        #print(current_score, params)
        return -current_score
    
    def hyperopt_lgb_score(params):
        clf = LGBMClassifier(**params)
        # усреднение по 3ем фолдам, для уменьшения влияния стахостичности
        # для ускорения можно использовать train_test_split один раз
        current_score = cross_val_score(clf, X, y,scoring=metric).mean()
        #print(current_score, params)
        return -current_score
    
    def hyperopt_ctb_score(params):
        clf = CatBoostClassifier(**params)
        # усреднение по 3ем фолдам, для уменьшения влияния стахостичности
        # для ускорения можно использовать train_test_split один раз
        current_score = cross_val_score(clf, X, y,scoring=metric).mean()
        #print(current_score, params)
        return -current_score
    
    def hyperopt_rf_score(params):
        clf = RandomForestClassifier(**params)
        # усреднение по 3ем фолдам, для уменьшения влияния стахостичности
        # для ускорения можно использовать train_test_split один раз
        current_score = cross_val_score(clf, X, y,scoring=metric).mean()
        #print(current_score, params)
        return -current_score
    
    if model == 'xgb':
        space = {
            'max_depth': hp.choice('max_depth', np.arange(3,13,1)),
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'lambda': hp.uniform('lambda', 0.0, 1.0),
            'gamma': hp.uniform('gamma', 0.0, 1.0),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1,13,1)),
            'subsample': hp.uniform('subsample', 0.3, 1.0),
            'n_estimators': hp.choice('n_estimators', np.arange(100, max_iterations,25)),
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor'
        }
        space.update(dict_concat)
        best = fmin(fn=hyperopt_xgb_score, space=space, algo=tpe.suggest, max_evals=evals)
        best['max_depth'] = np.arange(3,13,1)[best['max_depth']]
        best['min_child_weight'] = np.arange(1,13,1)[best['min_child_weight']]
        best['n_estimators'] = np.arange(100,max_iterations,25)[best['n_estimators']]
        res = XGBClassifier(**best)
        res.fit(X,y)
        
    if model == 'lgb':
        space = {
            'max_depth': hp.choice('max_depth', np.arange(3,13,1)),
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'num_leaves': hp.choice('num_leaves', np.arange(20,201,5)),
            'lambda': hp.uniform('lambda', 0.0, 1.0),
            'gamma': hp.uniform('gamma', 0.0, 1.0),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1,13,1)),
            'subsample': hp.uniform('subsample', 0.3, 1.0),
            'n_estimators': hp.choice('n_estimators', range(100, max_iterations,25)),
        }
        space.update(dict_concat)
        best = fmin(fn=hyperopt_lgb_score, space=space, algo=tpe.suggest, max_evals=evals)
        best['max_depth'] = np.arange(3,13,1)[best['max_depth']]
        best['num_leaves'] = np.arange(20,201,5)[best['num_leaves']]
        best['min_child_weight'] = np.arange(1,13,1)[best['min_child_weight']]
        best['n_estimators'] = np.arange(100,max_iterations,25)[best['n_estimators']]
        res = LGBMClassifier(**best)
        res.fit(X,y)
        
    if model == 'ctb':
        space = {
            'max_depth': hp.choice('max_depth', np.arange(3,13,1)),
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1.0),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.0, 1.0),
            'n_estimators': hp.choice('n_estimators', range(100, max_iterations,25)),
            'verbose': 0
        }
        space.update(dict_concat)
        best = fmin(fn=hyperopt_ctb_score, space=space, algo=tpe.suggest, max_evals=evals)
        best['max_depth'] = np.arange(3,13,1)[best['max_depth']]
        best['n_estimators'] = np.arange(100,max_iterations,25)[best['n_estimators']]
        res = CatBoostClassifier(**best)
        res.fit(X,y,verbose=False)
        
    if model == 'rf':
        space = {
             'bootstrap': hp.choice('bootstrap', [True, False]),
             'max_depth': hp.choice('max_depth', np.arange(5,50,1)),
             'max_features': hp.choice('max_features', ['auto', 'sqrt']),
             'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1,5,1)),
             'min_samples_split': hp.choice('min_samples_split', np.arange(2,10,2)),
             'n_estimators': hp.choice('n_estimators', range(100, max_iterations,25)),
             'verbose': 0,
        }
        space.update(dict_concat)
        best = fmin(fn=hyperopt_rf_score, space=space, algo=tpe.suggest, max_evals=evals)
        best['bootstrap'] = [True, False][best['bootstrap']]
        best['max_depth'] = np.arange(5,50,1)[best['max_depth']]
        best['max_features'] = ['auto', 'sqrt'][best['max_features']]
        best['min_samples_leaf'] = np.arange(1,5,1)[best['min_samples_leaf']]
        best['min_samples_split'] = np.arange(2,10,2)[best['min_samples_split']]
        best['n_estimators'] = np.arange(100,max_iterations,25)[best['n_estimators']]
        res = RandomForestClassifier(**best)
        res.fit(X,y)
        
    return res
    