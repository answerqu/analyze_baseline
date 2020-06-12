import pandas as pd
import numpy as np
import random

def data_preprocessing(data,encoder=None,scaler=None,random_state=random.randrange(1,100),
                       fill_nan_cat=None,fill_nan_num=None,
                       feature_exceptions_fill=[],feature_exceptions_encode=[],feature_exceptions_dummies=[],
                       features_list=[],scale_count=False,
                       dummies=False,drop_first=False,
                       drop_col=None):

    """
        Args:
     data - DataFrame to preprocess
     encoder - encode method from sklearn.preprocessing 
     scaler - scale method from sklearn.preprocessing
     random_state - not important param for multinomial distribution
     fill_nan_cat, fill_nan_num - =some_value - method for categorical and numerical columns
                                  ='top' - top feature for categorical columns
                                  ='multinomial' - multinomial disrtibution for empty spaces in each column 
                                  ='median' or ='mean' - fill numerical NaN by mean or median
                                                   
     feature_exceptions_fill,feature_exceptions_encode,feature_exceptions_dummies - columns to not to fill|encode|dummie
                                                                                    =['feature1','feature2',...]
     features_list - features to scale by group
                      =[['feature11','feature12'],['feature21','feature22'],...]
     dummies, drop_first - pd.get_dummies(df); 
                          by default: both == False
     drop_col - columns to drop 
     
        returns:
        preprocessed dataframe"
    """
    
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
            data[feature] = encoder.fit_transform(data[feature],data[feature])
    
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