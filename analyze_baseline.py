import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from scipy.stats import uniform,expon

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


from sklearn import preprocessing 
from sklearn.metrics import log_loss,roc_auc_score,f1_score,precision_score,recall_score,confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV,validation_curve, train_test_split
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK,STATUS_FAIL





import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


#############################################                  #################################################################
###########################################  DATA PREPROCESSING  ############################################################### 
#############################################                  #################################################################

# data - DataFrame to preprocess
# encoder - encode method from sklearn.preprocessing 
# scaler - scale method from sklearn.preprocessing
# random_state - not important param for multinomial distribution
# fill_nan_cat, fill_nan_num - =some_value - method for categorical and numerical columns
#                              ='top' - top feature for categorical columns
#                              ='multinomial' - multinomial disrtibution for empty spaces in each column 
#                              ='median' or ='mean' - fill numerical NaN by mean or median
#                                               
# feature_exceptions_fill,feature_exceptions_encode,feature_exceptions_dummies - columns to not to fill|encode|dummie
#                                                                                =['feature1','feature2',...]
# features_list - features to scale by group
#                  =[['feature11','feature12'],['feature21','feature22'],...]
# dummies, drop_first - pd.get_dummies(df); 
#                       by default: both == False
# drop_col - columns to drop 


def data_preprocessing(data,encoder=None,scaler=None,random_state=random.randrange(1,100),
                       fill_nan_cat=None,fill_nan_num=None,
                       feature_exceptions_fill=[],feature_exceptions_encode=[],feature_exceptions_dummies=[],
                       features_list=[],scale_count=False,
                       dummies=False,drop_first=False,
                       drop_col=None):
    cat_col = list(data.select_dtypes(include=['object']))
    num_col = list(data.select_dtypes(include=['float64', 'int64']))
    
    if drop_col != None:
        if drop_col == 'objective':
            data.drop(cat_col,axis=1,inplace=True)
        elif drop_col == 'numeric':
            data.drop(num_col,axis=1,inplace=True) 
        else:
            data.drop(drop_col,axis=1,inplace=True)
        cat_col = list(data.select_dtypes(include=['object']))
        num_col = list(data.select_dtypes(include=['float64', 'int64']))
            
    
    if fill_nan_cat != None:
        cat_col_fill = cat_col
        for feature in feature_exceptions_fill:
            cat_col_fill.remove(feature)
        if fill_nan_cat == 'top':
            for feature in cat_col_fill:
                data[feature] = data[feature].fillna(data[feature].describe().top)
        if fill_nan_cat == 'multinomial':
            func = lambda x: np.argmax(x)
            for feature in cat_col_fill:
                if data[feature].isna().sum() != 0: 
                    data_nan_index = list(data[data[feature].isna() == True].index)
                    num_tf = data[feature].isna().value_counts()
                    prob_list = data[feature].value_counts().values/num_tf[0]
                    indexes = data[feature].value_counts().index
                    data_new = pd.Series(map(func, multinomial.rvs(n=1,p=prob_list,size=num_tf[1],random_state=random_state)), 
                                             index = data_nan_index).apply(lambda x: indexes[x])
                    data.iloc[data_nan_index, list(data.columns).index(feature)] = data_new
        else:
            data[cat_col_fill] = data[cat_col_fill].fillna(fill_nan_cat)
                                         
    if fill_nan_num != None:
        num_col_fill = num_col
        for feature in feature_exceptions_fill:
            num_col_fill.remove(feature)                           
        if fill_nan_num == 'mean':
            for feature in num_col_fill:
                data[feature] = data[feature].fillna(data[feature].mean())
        if fill_nan_num == 'median':
            for feature in num_col_fill:
                data[feature] = data[feature].fillna(data[feature].median())
        else:
            data[num_col_fill] = data[num_col_fill].fillna(fill_nan_num)
                                         
    if encoder != None:
        cat_col_encode = cat_col
        for feature in feature_exceptions_encode:
            cat_col_encode.remove(feature)
        for feature in cat_col_encode:
            data[feature] = encoder.fit_transform(data[feature])
    
    if dummies:
        cat_col_dummies = cat_col
        for feature in feature_exceptions_dummies:
            cat_col_dummies.remove(feature)
        data = pd.get_dummies(data,drop_first=drop_first,columns = cat_col_dummies)
    
    if scaler != None:
        if scale_count:
            for features_scale in features_list:
                data_array = np.transpose(scaler.fit_transform(data[features_scale]))
                for i in range(len(features_scale)):
                    data[features_scale[i]] = data_array[i]
        else:
            data_array = np.transpose(scaler.fit_transform(data[features_list]))
            for i in range(len(features_list)):
                data[features_list[i]] = data_array[i]
    return data

#############################################              ####################################################################
###########################################  MODEL TRAINING  ################################################################## 
#############################################              ####################################################################

# X,y -data,target
# test_size - test size for train_test_split()
# task_type = 'classification' or 'regression', for now only 'classification' available
# score - type of scoring to raise
# CCV - if True: make a cross validation if False: don't
# cv - number of folds for CCV
# score_matrix_print - if True: print score dataframe with models for columns and scorings for indexes
# alg_list_clf_names - list of algorithms names (using for printing info)
# alg_list_clf - list of sklearn models
# params_distribution - list of dictionaries of parameters distribution of each model for GridSearchCV; 
# plot - if True: plot validation curve for each parameter in params_distribution



n_iters = 50
random_state = 42

alg_list_clf= [ DecisionTreeClassifier(random_state= random_state),
               LogisticRegression(),
                GaussianNB(),
            SGDClassifier(max_iter= 200,random_state= random_state),
            KNeighborsClassifier(),
            SVC()
            ]

alg_list_clf_names = ['Decision Tree Classifier','Logistic regression', 'Naive Bayes', 'SGD', 'kNN',  'Support Vector Machines (SVC)']

params_distribution=[{'max_depth': range(4,13),
                      'min_samples_split': np.arange(2,41,5),
                      'min_samples_leaf': np.arange(2,21,5),

                     }, 
    
                    #logreg
                     {'C': np.logspace(-3,3,n_iters),
                      #'solver': ['liblinear','lbfgs',***],
                      #'class_weight':
                      #'penalty': ['l1','l2','elasticnet'],

                     },
                    
                     #NB
                      {},                   
                      
                    #SGD
                     {'alpha': np.logspace(-3, 3, n_iters), # learning rate
                      
                      #'loss': ['log','hinge','modified_huber'],
                      #'penalty': ['l2','l1','elasticnet'],
  
                     },
    
                    #kNN
                    {'n_neighbors': range(1,12),   
                    },
                    #tree
                     
    
                    #SVM
                    {'C': np.logspace(-3,3,n_iters),
                     }]

def model_train_light(X,y,test_size=0.25,task_type='classification',score='roc_auc',CCV=False,cv=3,score_matrix_print=True,
                      random_state=random_state,alg_list_clf_names = alg_list_clf_names,
                      alg_list_clf=alg_list_clf,params_distribution=params_distribution,
                      plot=False):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size,random_state=random_state,shuffle=True)
    list_of_models = []
    score_matrix = []
    if task_type == 'classification': #lr,B, SGD, kNN, tree, SVM, RFM
        for light_clf in alg_list_clf:
            if (alg_list_clf.index(light_clf) != -1):
                curr_index = alg_list_clf.index(light_clf)                    
                if CCV:
                    curr_grid_model = GridSearchCV(light_clf,param_grid=params_distribution[curr_index],
                                                  scoring=score,cv=cv)
                    curr_grid_model.fit(X,y)
                    print('MODEL: ', alg_list_clf_names[curr_index])
                    print('Best score on CCV: ', curr_grid_model.best_score_)
                    print('Best parameters on CCV: ', curr_grid_model.best_params_)
                    print()
                else:
                    curr_grid_model = light_clf
                    print('MODEL: ', alg_list_clf_names[curr_index])
                    print('Best score on CCV: ', curr_grid_model.best_score_)
                    print('Best parameters on CCV: ', curr_grid_model.best_params_)
                    print()
                if plot:
                    lw=2
                    for param in list(curr_grid_model.best_params_.keys()):
                        param_range = params_distribution[curr_index][param]
                        train_scores, test_scores = validation_curve(light_clf, X, y, param_name=param,
                                                                         param_range=param_range,scoring=score)
                        train_scores_mean = np.mean(train_scores, axis=1)
                        train_scores_std = np.std(train_scores, axis=1)
                        test_scores_mean = np.mean(test_scores, axis=1)
                        test_scores_std = np.std(test_scores, axis=1)
                        plt.figure(figsize=(12,8))
                        plt.title('Validation Curve for ' + str(alg_list_clf_names[curr_index]))
                        plt.xlabel(param)
                        plt.ylabel(str(score) + " score")
                        plt.semilogx(param_range, train_scores_mean, label="Training score",
                                         color="darkorange", lw=lw)
                        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                                             train_scores_mean + train_scores_std, alpha=0.2,
                                             color="darkorange", lw=lw)
                        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                                         color="navy", lw=lw)
                        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                                             test_scores_mean + test_scores_std, alpha=0.2,
                                             color="navy", lw=lw)
                        plt.legend(loc="best")
                        plt.show()
                if score_matrix_print:
                    score_array = np.array([accuracy_score(y_test,curr_grid_model.predict(X_test)),
                                            precision_score(y_test,curr_grid_model.predict(X_test)),
                                            recall_score(y_test,curr_grid_model.predict(X_test)),
                                            f1_score(y_test,curr_grid_model.predict(X_test)),
                                            roc_auc_score(y_test,curr_grid_model.predict(X_test))],dtype='float64')
                    score_matrix.append(score_array)
                    print(confusion_matrix(y_test,curr_grid_model.predict(X_test))) 
                    print()
                    print('#####################')
                    print()
                list_of_models.append(curr_grid_model)
    if score_matrix:
        scoring=['accuracy','precicion', 'recall', 'f1', 'roc_auc_score']
        score_df = pd.DataFrame(score_matrix, columns=scoring,index = alg_list_clf_names)
    return score_df, list_of_models
