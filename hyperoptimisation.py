import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
import sklearn.ensemble as ens

from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from hyperopt import hp

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_absolute_error

def hp_model_classification(X_train,y_train,X_eval,y_eval,X_test=None,y_test=None, 
             max_evals=10,cat_features=[], metric=f1_score, n_estimators=500,
             early_stopping_rounds=50,
             model=xgb):
    if model = ens:
            
        rf_clf_params = {
                                'bootstrap': hp.choice('bootstrap', [True, False]),
                                'max_depth': hp.choice('max_depth', np.arange(5,50,1)),
                                'max_features': hp.choice('max_features', ['auto', 'sqrt']),
                                'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1,5,1)),
                                'min_samples_split': hp.choice('min_samples_split', np.arange(2,10,2)),
                                'n_estimators':     hp.choice('n_estimators', np.arange(100,1500,100)),
                                'verbose': 0,
        }
        rf_fit_params = {
                #'early_stopping_rounds': 150,
                #'verbose': False,
        }
        rf_para = dict()
        rf_para['clf_params'] = rf_clf_params
        rf_para['fit_params'] = rf_fit_params
        rf_para['metric' ] = lambda y, pred: -metric(y,pred)
        
    if model = xgb:
        xgb_clf_params = {
                                'max_depth': hp.choice('max_depth', np.arange(3,13,1)),
                                'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
                                'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
                                'num_leaves': hp.choice('num_leaves', np.arange(20,201,5)),
                                'lambda': hp.uniform('lambda', 0.0, 1.0),
                                'gamma': hp.uniform('gamma', 0.0, 1.0),
                                'min_child_weight': hp.choice('min_child_weight', np.arange(1,13,1)),
                                'subsample': hp.uniform('subsample', 0.3, 1.0),
                                'n_estimators':     n_estimators,
        }
        xgb_fit_params = {
                'early_stopping_rounds': early_stopping_rounds,
                'verbose': False
        }
        xgb_para = dict()
        xgb_para['clf_params'] = xgb_clf_params
        xgb_para['fit_params'] = xgb_fit_params
        xgb_para['metric' ] = lambda y, pred: -metric(y,pred)
    
    
    if model = lgb:
        lgb_clf_params = {
                                'max_depth': hp.choice('max_depth', np.arange(3,13,1)),
                                'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
                                'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
                                'num_leaves': hp.choice('num_leaves', np.arange(20,201,5)),
                                'lambda': hp.uniform('lambda', 0.0, 1.0),
                                'gamma': hp.uniform('gamma', 0.0, 1.0),
                                'min_child_weight': hp.choice('min_child_weight', np.arange(1,13,1)),
                                'subsample': hp.uniform('subsample', 0.3, 1.0),
                                'n_estimators':     n_estimators,
        }
        lgb_fit_params = {
                'early_stopping_rounds': early_stopping_rounds,
                'verbose': False
        }
        lgb_para = dict()
        lgb_para['clf_params'] = lgb_clf_params
        lgb_para['fit_params'] = lgb_fit_params
        lgb_para['metric' ] = lambda y, pred: -metric(y,pred)
    
    
    if model = ctb:
        ctb_clf_params = {
                                'max_depth': hp.choice('max_depth', np.arange(3,13,1)),
                                'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
                                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1.0),
                                #'num_leaves': hp.choice('num_leaves', np.arange(20,201,20)),
                                'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.0, 1.0),
                                #'gamma': hp.uniform('gamma', 0.0, 1.0),
                                #'min_child_samples': hp.choice('min_child_samples', np.arange(1,13,1)),
                                'subsample': hp.uniform('subsample', 0.3, 1.0),
                                'n_estimators':     n_estimators,
        }
        ctb_fit_params = {
                'early_stopping_rounds': early_stopping_rounds,
                'verbose': False,
                'cat_features': cat_features
        }
        ctb_para = dict()
        ctb_para['clf_params'] = ctb_clf_params
        ctb_para['fit_params'] = ctb_fit_params
        ctb_para['metric' ] = lambda y, pred: -metric(y,pred)
    
    class HPOpt(object):
    
        def __init__(self, x_train, x_test, y_train, y_test):
            self.x_train = x_train
            self.x_test  = x_test
            self.y_train = y_train
            self.y_test  = y_test
    
        def process(self, fn_name, space, trials, algo, max_evals):
            fn = getattr(self, fn_name)
            try:
                result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
            except Exception as e:
                return {'status': STATUS_FAIL,
                        'exception': str(e)}
            return result, trials
        
        def rf_clf(self, para):
            clf = ens.RandomForestClassifier(**para['clf_params'])
            clf.fit(self.x_train, self.y_train,
                    **para['fit_params'])
            pred = clf.predict(self.x_test)
            loss = para['metric'](self.y_test, pred)
            return {'score': loss, 'status': STATUS_OK}
        
        
        def xgb_clf(self, para):
            clf = xgb.XGBClassifier(**para['clf_params'])
            return self.train_clf(clf, para)
    
        def lgb_clf(self, para):
            clf = lgb.LGBMClassifier(**para['clf_params'])
            return self.train_clf(clf, para)
    
        def ctb_clf(self, para):
            clf = ctb.CatBoostClassifier(**para['clf_params'])
            return self.train_clf(clf, para)
    
        def train_clf(self, clf, para):
            clf.fit(self.x_train, self.y_train,
                    eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                    **para['fit_params'])
            pred = clf.predict(self.x_test)
            loss = para['metric'](self.y_test, pred)
            return {'score': loss, 'status': STATUS_OK}
    
    if len(cat_features) != 0:
        obj_cat = HPOpt(X_train, X_eval, y_train, y_eval)
        
        le = LabelEncoder()
        ind_slice1 = X_train.shape[0]
        ind_slice2 = X_eval.shape[0] 
        X_concat = pd.concat([X_train,X_eval,X_test])
        for a in cat_features:
            X_concat[a] = le.fit_transform(X_concat[a])
        X_train , X_eval, X_test = X_concat.iloc[:ind_slice1,:], X_concat.iloc[ind_slice1:ind_slice1+ind_slice2,:],X_concat.iloc[ind_slice1+ind_slice2:,:]
        obj = HPOpt(X_train, X_eval, y_train, y_eval)
    else:
        obj = HPOpt(X_train, X_eval, y_train, y_eval)
        obj_cat = HPOpt(X_train, X_eval, y_train, y_eval)
        
    
    
    print('RandomForestClassifier hyperoptimisation in progress...')
    rf_opt = obj.process(fn_name='rf_clf', space=rf_para, trials=Trials(), algo=tpe.suggest, max_evals=max_evals/2)
    rf_opt[0]['bootstrap'] = [True, False][rf_opt[0]['bootstrap']]
    rf_opt[0]['max_depth'] = np.arange(5,50,1)[rf_opt[0]['max_depth']]
    rf_opt[0]['max_features'] = ['auto', 'sqrt'][rf_opt[0]['max_features']]
    rf_opt[0]['min_samples_leaf'] = np.arange(1,5,1)[rf_opt[0]['min_samples_leaf']]
    rf_opt[0]['min_samples_split'] = np.arange(2,10,2)[rf_opt[0]['min_samples_split']]
    rf_opt[0]['n_estimators'] = np.arange(100,1500,100)[rf_opt[0]['n_estimators']]
    
    
    print('XGBClassifier hyperoptimisation in progress...')
    xgb_opt = obj.process(fn_name='xgb_clf', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=10*max_evals)
    xgb_opt[0]['max_depth'] = np.arange(3,13,1)[xgb_opt[0]['max_depth']]
    xgb_opt[0]['num_leaves'] = np.arange(20,201,5)[xgb_opt[0]['num_leaves']]
    xgb_opt[0]['min_child_weight'] = np.arange(1,13,1)[xgb_opt[0]['min_child_weight']]
    
    print('LGBMClassifier hyperoptimisation in progress...')
    lgb_opt = obj.process(fn_name='lgb_clf', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=10*max_evals)
    lgb_opt[0]['max_depth'] = np.arange(3,13,1)[lgb_opt[0]['max_depth']]
    lgb_opt[0]['num_leaves'] = np.arange(20,201,5)[lgb_opt[0]['num_leaves']]
    lgb_opt[0]['min_child_weight'] = np.arange(1,13,1)[lgb_opt[0]['min_child_weight']]
   
    print('CatBoostClassifier hyperoptimisation in progress...')
    ctb_opt = obj_cat.process(fn_name='ctb_clf', space=ctb_para, trials=Trials(), algo=tpe.suggest, max_evals=round(max_evals/3))
    ctb_opt[0]['max_depth'] = np.arange(3,13,1)[ctb_opt[0]['max_depth']]
    
    return rf_opt, xgb_opt,lgb_opt,ctb_opt

def hp_model_regression(X_train,y_train,X_eval,y_eval,X_test=None,y_test=None, 
             max_evals=10,cat_features=[], metric=mean_absolute_error, n_estimators=500,
             early_stopping_rounds=50,
             model=xgb):
    if model = ens:
            
        rf_reg_params = {
                                'bootstrap': hp.choice('bootstrap', [True, False]),
                                'max_depth': hp.choice('max_depth', np.arange(5,50,1)),
                                'max_features': hp.choice('max_features', ['auto', 'sqrt']),
                                'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1,5,1)),
                                'min_samples_split': hp.choice('min_samples_split', np.arange(2,10,2)),
                                'n_estimators':     hp.choice('n_estimators', np.arange(100,1500,100)),
                                'verbose': 0,
        }
        rf_fit_params = {
                #'early_stopping_rounds': 150,
                #'verbose': False,
        }
        rf_para = dict()
        rf_para['reg_params'] = rf_reg_params
        rf_para['fit_params'] = rf_fit_params
        rf_para['metric' ] = lambda y, pred: -metric(y,pred)
        
    if model = xgb:
        xgb_reg_params = {
                                'max_depth': hp.choice('max_depth', np.arange(3,13,1)),
                                'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
                                'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
                                'num_leaves': hp.choice('num_leaves', np.arange(20,201,5)),
                                'lambda': hp.uniform('lambda', 0.0, 1.0),
                                'gamma': hp.uniform('gamma', 0.0, 1.0),
                                'min_child_weight': hp.choice('min_child_weight', np.arange(1,13,1)),
                                'subsample': hp.uniform('subsample', 0.3, 1.0),
                                'n_estimators':     n_estimators,
        }
        xgb_fit_params = {
                'early_stopping_rounds': early_stopping_rounds,
                'verbose': False
        }
        xgb_para = dict()
        xgb_para['reg_params'] = xgb_reg_params
        xgb_para['fit_params'] = xgb_fit_params
        xgb_para['metric' ] = lambda y, pred: -metric(y,pred)
    
    
    if model = lgb:
        lgb_reg_params = {
                                'max_depth': hp.choice('max_depth', np.arange(3,13,1)),
                                'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
                                'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
                                'num_leaves': hp.choice('num_leaves', np.arange(20,201,5)),
                                'lambda': hp.uniform('lambda', 0.0, 1.0),
                                'gamma': hp.uniform('gamma', 0.0, 1.0),
                                'min_child_weight': hp.choice('min_child_weight', np.arange(1,13,1)),
                                'subsample': hp.uniform('subsample', 0.3, 1.0),
                                'n_estimators':     n_estimators,
        }
        lgb_reg_params = {
                'early_stopping_rounds': early_stopping_rounds,
                'verbose': False
        }
        lgb_para = dict()
        lgb_para['reg_params'] = lgb_reg_params
        lgb_para['fit_params'] = lgb_fit_params
        lgb_para['metric' ] = lambda y, pred: -metric(y,pred)
    
    
    if model = ctb:
        ctb_reg_params = {
                                'max_depth': hp.choice('max_depth', np.arange(3,13,1)),
                                'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
                                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1.0),
                                #'num_leaves': hp.choice('num_leaves', np.arange(20,201,20)),
                                'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.0, 1.0),
                                #'gamma': hp.uniform('gamma', 0.0, 1.0),
                                #'min_child_samples': hp.choice('min_child_samples', np.arange(1,13,1)),
                                'subsample': hp.uniform('subsample', 0.3, 1.0),
                                'n_estimators':     n_estimators,
        }
        ctb_fit_params = {
                'early_stopping_rounds': early_stopping_rounds,
                'verbose': False,
                'cat_features': cat_features
        }
        ctb_para = dict()
        ctb_para['reg_params'] = ctb_reg_params
        ctb_para['fit_params'] = ctb_fit_params
        ctb_para['metric' ] = lambda y, pred: -metric(y,pred)
    
    class HPOpt(object):
    
        def __init__(self, x_train, x_test, y_train, y_test):
            self.x_train = x_train
            self.x_test  = x_test
            self.y_train = y_train
            self.y_test  = y_test
    
        def process(self, fn_name, space, trials, algo, max_evals):
            fn = getattr(self, fn_name)
            try:
                result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
            except Exception as e:
                return {'status': STATUS_FAIL,
                        'exception': str(e)}
            return result, trials
        
        def rf_reg(self, para):
            reg = ens.RandomForestRegressor(**para['reg_params'])
            reg.fit(self.x_train, self.y_train,
                    **para['fit_params'])
            pred = reg.predict(self.x_test)
            loss = para['metric'](self.y_test, pred)
            return {'score': loss, 'status': STATUS_OK}
        
        
        def xgb_reg(self, para):
            reg = xgb.XGBRegressor(**para['reg_params'])
            return self.train_reg(reg, para)
    
        def lgb_reg(self, para):
            reg = lgb.LGBMRegressor(**para['reg_params'])
            return self.train_reg(reg, para)
    
        def ctb_reg(self, para):
            reg = ctb.CatBoostRegressor(**para['reg_params'])
            return self.train_reg(reg, para)
    
        def train_reg(self, reg, para):
            reg.fit(self.x_train, self.y_train,
                    eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                    **para['fit_params'])
            pred = reg.predict(self.x_test)
            loss = para['metric'](self.y_test, pred)
            return {'score': loss, 'status': STATUS_OK}
    
    if len(cat_features) != 0:
        obj_cat = HPOpt(X_train, X_eval, y_train, y_eval)
        
        le = LabelEncoder()
        ind_slice1 = X_train.shape[0]
        ind_slice2 = X_eval.shape[0] 
        X_concat = pd.concat([X_train,X_eval,X_test])
        for a in cat_features:
            X_concat[a] = le.fit_transform(X_concat[a])
        X_train , X_eval, X_test = X_concat.iloc[:ind_slice1,:], X_concat.iloc[ind_slice1:ind_slice1+ind_slice2,:],X_concat.iloc[ind_slice1+ind_slice2:,:]
        obj = HPOpt(X_train, X_eval, y_train, y_eval)
    else:
        obj = HPOpt(X_train, X_eval, y_train, y_eval)
        obj_cat = HPOpt(X_train, X_eval, y_train, y_eval)
        
    
    
    print('RandomForestClassifier hyperoptimisation in progress...')
    rf_opt = obj.process(fn_name='rf_reg', space=rf_para, trials=Trials(), algo=tpe.suggest, max_evals=max_evals/2)
    rf_opt[0]['bootstrap'] = [True, False][rf_opt[0]['bootstrap']]
    rf_opt[0]['max_depth'] = np.arange(5,50,1)[rf_opt[0]['max_depth']]
    rf_opt[0]['max_features'] = ['auto', 'sqrt'][rf_opt[0]['max_features']]
    rf_opt[0]['min_samples_leaf'] = np.arange(1,5,1)[rf_opt[0]['min_samples_leaf']]
    rf_opt[0]['min_samples_split'] = np.arange(2,10,2)[rf_opt[0]['min_samples_split']]
    rf_opt[0]['n_estimators'] = np.arange(100,1500,100)[rf_opt[0]['n_estimators']]
    
    
    print('XGBClassifier hyperoptimisation in progress...')
    xgb_opt = obj.process(fn_name='xgb_reg', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=10*max_evals)
    xgb_opt[0]['max_depth'] = np.arange(3,13,1)[xgb_opt[0]['max_depth']]
    xgb_opt[0]['num_leaves'] = np.arange(20,201,5)[xgb_opt[0]['num_leaves']]
    xgb_opt[0]['min_child_weight'] = np.arange(1,13,1)[xgb_opt[0]['min_child_weight']]
    
    print('LGBMClassifier hyperoptimisation in progress...')
    lgb_opt = obj.process(fn_name='lgb_reg', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=10*max_evals)
    lgb_opt[0]['max_depth'] = np.arange(3,13,1)[lgb_opt[0]['max_depth']]
    lgb_opt[0]['num_leaves'] = np.arange(20,201,5)[lgb_opt[0]['num_leaves']]
    lgb_opt[0]['min_child_weight'] = np.arange(1,13,1)[lgb_opt[0]['min_child_weight']]
   
    print('CatBoostClassifier hyperoptimisation in progress...')
    ctb_opt = obj_cat.process(fn_name='ctb_reg', space=ctb_para, trials=Trials(), algo=tpe.suggest, max_evals=round(max_evals/3))
    ctb_opt[0]['max_depth'] = np.arange(3,13,1)[ctb_opt[0]['max_depth']]
    
    return rf_opt, xgb_opt,lgb_opt,ctb_opt