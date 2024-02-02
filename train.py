from pyspark import SparkContext
import time
import json
import csv
import pandas as pd
import numpy as np
import joblib
import sys

from sklearn.model_selection import GridSearchCV as xgb_gridsearchcv
from xgboost import XGBRegressor

from sklearn.decomposition import PCA
from datetime import datetime
from collections import Counter

from surprise import Dataset, Reader, SVD
from surprise.model_selection.search import GridSearchCV as svd_gridsearchcv

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
from statistics import mean

folder_path =  sys.argv[1] # ./data
test_file_path = sys.argv[2] # ./data/yelp_val.csv
output_file_path = sys.argv[3] # ./output.csv

sc = SparkContext('local[*]','Train')

# step 1: Data Processing
train_rdd = sc.textFile(folder_path + '/yelp_train.csv').filter(lambda line: line != 'user_id,business_id,stars')\
    .map(lambda line: line.split(',')).cache()
# test_rdd = sc.textFile(test_file_path).filter(lambda line: line != 'user_id,business_id,stars')\
#     .map(lambda line: line.split(',')).cache()
print('step 1 finish')


# step 2: Feacture extraction for user
user_dict_base = sc.textFile(folder_path + '/user.json').map(lambda line: json.loads(line))\
                                       .map(lambda line:(line['user_id'],(
                                       (datetime.now() - datetime.strptime(line['yelping_since'], "%Y-%m-%d")).days,
                                       float(line['review_count']),
                                       float(line['average_stars']),
                                       float(line['fans'])**2,
                                       float(len(line.get('friends',[]).split(',') if line.get('friends') else []))**2,
                                        float(line['useful']),
                                        float(line['funny']),
                                        float(line['cool'])
                                       ))).collectAsMap()
print('step 2 finish')


# step 3: Feacture extraction for business
business_rdd = sc.textFile(folder_path + '/business.json').map(lambda line: json.loads(line)).cache()

# dict: {business_id: (average_star, review_count)}
business_dict_base = business_rdd.map(lambda business: (business['business_id'], (float(business['stars']),
                                                                                 float(business['review_count']),
                                                                                 business.get('latitude', None),
                                                                                 business.get('longitude', None))))\
    .collectAsMap()

# dict: {business_id: (dummy_state, dummy_city, dummy_categories)}
all_cities = business_rdd.map(lambda business: business['city']).collect()
cities_counts = Counter(all_cities)
unique_cities_100 = [city for city, count in cities_counts.items() if count >= 100]
unique_cities_bc = sc.broadcast(unique_cities_100)
unique_states = business_rdd.map(lambda business: business['state']).distinct().collect()
unique_states_bc = sc.broadcast(unique_states)
all_category = business_rdd.map(lambda business: business.get('categories', '').split(',') if business.get('categories') else [])\
    .flatMap(lambda categories: list(filter(None, categories))).collect()
category_counts = Counter(all_category)
unique_category_200 = [category for category, count in category_counts.items() if count >= 200]
unique_category_bc = sc.broadcast(unique_category_200)
print('step 3 finish')


# step 4: Principal Component Analysis
def one_hot_encode(business):
    business_city = business[0]
    business_state = business[1]
    if business[2]:
        business_category = list(filter(None,business[2].split(',')))
    else:
        business_category = []

    dummy_state = [1 if business_city == city else 0 for city in unique_cities_bc.value]
    dummy_city = [1 if business_state == state else 0 for state in unique_states_bc.value]

    if business_category == []:
        dummy_category = [0] * len(unique_category_bc.value)
    else:
        dummy_category = [1 if category in business_category else 0 for category in unique_category_bc.value]
    return tuple(dummy_city), tuple(dummy_state), tuple(dummy_category)

business_dummy_rdd = business_rdd.map(lambda business: (business['city'], business['state'], business['categories'] if business.get('categories') else None))\
    .map(one_hot_encode)

dummy_city = np.array(business_dummy_rdd.map(lambda business: business[0]).collect())
dummy_state = np.array(business_dummy_rdd.map(lambda business: business[1]).collect())
dummy_category = np.array(business_dummy_rdd.map(lambda business: business[2]).collect())

pca_city = PCA(n_components = 10).fit_transform(dummy_city)
pca_state = PCA(n_components = 10).fit_transform(dummy_state)
pca_category = PCA(n_components = 10).fit_transform(dummy_category)

business_dummy_dict = {key: (pca_city[i]).tolist()+(pca_state[i]).tolist()+(pca_category[i]).tolist() for i, key in enumerate(business_dict_base.keys())}
print('step 4 finish')


# step 5: Save result
model_folder = './model'
joblib.dump({'business_model_base': business_dict_base} , f'{model_folder}/business_model_base.joblib')
joblib.dump({'business_model_pca': business_dummy_dict} , f'{model_folder}/business_model_pca.joblib')
print('step 5 finish')


# step 6: Load result
# business_dict_base = joblib.load(f'{model_folder}/business_model_base.joblib')['business_model_base']
# business_dummy_dict = joblib.load(f'{model_folder}/business_model_pca.joblib')['business_model_pca']



# step 7: Model-based CF - Create xtrain, ytrain, xtest, ytrue
user_base_attrs_missing = [sum(values) / len(values) for values in zip(*user_dict_base.values())]
# user_social_attrs_missing = [sum(values) / len(values) for values in zip(*user_dict_social.values())]
business_base_attrs_missing = [sum(value for value in values if value is not None) / len([value for value in values if value is not None]) if values is not None else None for values in zip(*business_dict_base.values())]
business_dummy_attrs_missing = [sum(values) / len(values) for values in zip(*business_dummy_dict.values())]

def extract_features(line):
    user_id, business_id = line[0], line[1]
    user_bsse_attrs = user_dict_base.get(user_id,user_base_attrs_missing)
    # user_social_attrs = user_dict_social.get(user_id,user_social_attrs_missing)
    business_base_attrs = business_dict_base.get(business_id, business_base_attrs_missing)
    business_dummy_attrs = business_dummy_dict.get(business_id, business_dummy_attrs_missing)
    return list(user_bsse_attrs)+list(business_base_attrs)+list(business_dummy_attrs)

xtrain = np.array(train_rdd.map(lambda line: extract_features(line)).collect(), dtype='float32')
ytrain = np.array(train_rdd.map(lambda line: line[2]).collect(), dtype='float32')

# test_in_rdd = test_rdd.map(lambda line: (line[0],line[1])).cache()

# xtest = np.array(test_in_rdd.map(lambda line: extract_features(line)).collect(), dtype='float32')
# ytrue = np.array(test_rdd.map(lambda line: float(line[2])).collect(), dtype='float32')
# xtest_id = np.array(test_in_rdd.collect())
print('step 7 finish')


# step 8: Model-based CF - Hyperparameter tuning
# param_grid = {
#     'booster': ['gbtree'],
#     'max_depth': [3,5,7],
#     'min_child_weight': [1,3,5],
#     'random_state': [553],
#     'subsample':[0.8, 0.9, 1.0],
#     'colsample_bytree':[0.8, 0.9, 1.0],
#     'learning_rate': [0.01 ,0.1, 0.2],
#     'n_estimators': [100, 200, 500],
# }
# grid_search = xgb_gridsearchcv(estimator=XGBRegressor(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
# grid_search.fit(xtrain, ytrain)
# print("Best Hyperparameters:", grid_search.best_params_)
# xgb_model = grid_search.best_estimator_
# ytest = xgb_model.predict(xtest)
# ytrue = test_rdd.map(lambda line: float(line[2])).collect()
# rmse = sqrt(mean_squared_error(ytrue, (ytest).tolist()))
# print('RMSE', rmse)
'''
Result:
Fitting 5 folds for each of 729 candidates, totalling 3645 fits
Best Hyperparameters: {'booster': 'gbtree', 'colsample_bytree': 1, 'learning_rate': 0.1 ,'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 500, 'random_state': 553, 'subsample': 1.0}
RMSE 0.9787393340067615
'''
# Best Hyperparameters
xgb_param = {
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 1.0,
    'colsample_bytree': 1,
    'max_depth': 5,
    'min_child_weight': 5,
    'random_state': 553
}
# xgb_model = XGBRegressor(**xgb_param,booster="gbtree")
# xgb_model.fit(xtrain, ytrain)
# ytest = xgb_model.predict(xtest)
# ytrue = test_rdd.map(lambda line: float(line[2])).collect()
# rmse = sqrt(mean_squared_error(ytrue, (ytest).tolist()))
# print('RMSE', rmse)
print('step 8 finish')


# step 9: Model: SVD - Create train_df, test_in_df
train_df = pd.DataFrame(train_rdd.collect(),columns=['user_id', 'business_id', 'stars'])
# test_in_df = pd.DataFrame(test_rdd.collect(),columns=['user_id', 'business_id', 'stars'])

reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(train_df[['user_id', 'business_id', 'stars']], reader)
trainset = train_data.build_full_trainset()

# test_data = Dataset.load_from_df(test_in_df[['user_id', 'business_id', 'stars']], reader)
# testset = test_data.build_full_trainset().build_testset()
print('step 9 finish')


# step 10: Model: SVD - Hyperparameter tuning
# param_grid = {
#     'lr_all': [0.005],
#     'reg_all': [0.02, 0.1],
#     'n_factors': [2, 5, 10],
#     'n_epochs': [5, 20, 50],
# }
# grid_search = svd_gridsearchcv(
#     algo_class = SVD,
#     param_grid = param_grid,
#     n_jobs = -1,
#     joblib_verbose = 5,
#     cv = 5)
# grid_search.fit(train_data)
# print('Best parameters:', grid_search.best_params['rmse'])
# print('Best RMSE:', grid_search.best_score['rmse'])
# best_svd = grid_search.best_estimator['rmse']
# best_svd.fit(trainset)
# svd_pred_test = best_svd.test(testset)
# svd_pred_test_dict = {(prediction.uid, prediction.iid):prediction.est for prediction in svd_pred_test}
# svd_pred_test_r = []
# for prediction in test_rdd.map(lambda line: (line[0],line[1])).collect():
#             if svd_pred_test_dict.get(prediction):
#                 svd_pred_test_r.append(svd_pred_test_dict[prediction])
# ytrue = test_rdd.map(lambda line: float(line[2])).collect()
# rmse = sqrt(mean_squared_error(ytrue, svd_pred_test_r))
# print('RMSE', rmse)
'''
Result:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  14 tasks      | elapsed:   30.3s
[Parallel(n_jobs=-1)]: Done  68 tasks      | elapsed:  2.5min
[Parallel(n_jobs=-1)]: Done  90 out of  90 | elapsed:  3.5min finished
Best parameters: {'lr_all': 0.005, 'reg_all': 0.1, 'n_factors': 2, 'n_epochs': 50, 'random_state': 553}
Best RMSE: 1.011886321693489
RMSE 1.0025289981200516
'''
# Best Hyperparameters
svd_param = {
    'lr_all': 0.005,
    'reg_all': 0.1,
    'n_factors': 2,
    'n_epochs': 50,
    'random_state': 553
}
# svd_model = SVD(**svd_param)
# svd_model.fit(trainset)
# svd_pred_test = svd_model.test(testset)
# svd_pred_test_dict = {(prediction.uid, prediction.iid):prediction.est for prediction in svd_pred_test}
# svd_pred_test_r = []
# for prediction in test_rdd.map(lambda line: (line[0],line[1])).collect():
#             if svd_pred_test_dict.get(prediction):
#                 svd_pred_test_r.append(svd_pred_test_dict[prediction])

# ytrue = test_rdd.map(lambda line: float(line[2])).collect()
# rmse = sqrt(mean_squared_error(ytrue, svd_pred_test_r))
# print('RMSE', rmse)
print('step 10 finish')


# step 11: Bagging
kfold = KFold(n_splits = 10, random_state = 553, shuffle = True)

for i, (kfold_train_index, kfold_test_index) in enumerate(kfold.split(xtrain)):
    print(f'Round {i}')
    # Model: xgboost
    xtrain_k, ytrain_k = xtrain[kfold_train_index], ytrain[kfold_train_index]
    xval_k, yval_k = xtrain[kfold_test_index], ytrain[kfold_test_index]

    xgb_param = {
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 1.0,
    'colsample_bytree': 1,
    'max_depth': 5,
    'min_child_weight': 5,
    'random_state': 553
    }

    xgb_model = XGBRegressor(**xgb_param,booster="gbtree")
    xgb_model.fit(xtrain_k, ytrain_k)
    yval_pred_k = xgb_model.predict(xval_k)
    ytrain_pred_k = xgb_model.predict(xtrain_k)

    print('xgb_model Train RMSE:', sqrt(mean_squared_error(ytrain_k, (ytrain_pred_k).tolist())))
    print('xgb_model Validation RMSE:', sqrt(mean_squared_error(yval_k, (yval_pred_k).tolist())))

    # Model: SVD
    train_data_k = Dataset.load_from_df(train_df.iloc[kfold_train_index], reader)
    val_data_k = Dataset.load_from_df(train_df.iloc[kfold_test_index], reader)
    trainset_k = train_data_k.build_full_trainset()
    valset_k = val_data_k.build_full_trainset().build_testset()

    svd_param = {
    'lr_all': 0.005,
    'reg_all': 0.1,
    'n_factors': 2,
    'n_epochs': 50,
    'random_state': 553
    }

    svd_model = SVD(**svd_param)
    svd_model.fit(trainset_k)

    svd_pred_r = []
    svd_pred = svd_model.test(valset_k)
    svd_pred_dict = {(prediction.uid, prediction.iid):prediction.est for prediction in svd_pred}
    for prediction in train_rdd.map(lambda line: (line[0],line[1])).collect():
        if svd_pred_dict.get(prediction):
            svd_pred_r.append(svd_pred_dict[prediction])

    svd_train_pred_r = []
    svd_train_pred = svd_model.test(trainset_k.build_testset())
    svd_train_pred_dict = {(prediction.uid, prediction.iid):prediction.est for prediction in svd_train_pred}
    for prediction in train_rdd.map(lambda line: (line[0],line[1])).collect():
        if svd_train_pred_dict.get(prediction):
            svd_train_pred_r.append(svd_train_pred_dict[prediction])

    print('svd_model Train RMSE:', sqrt(mean_squared_error(ytrain_k, svd_train_pred_r)))
    print('svd_model Validation RMSE:', sqrt(mean_squared_error(yval_k, svd_pred_r)))

    joblib.dump({'xgb_model': xgb_model, 'svd_model':svd_model},f'{model_folder}/model_{i}.joblib')

'''
Result:
Round 0
xgb_model Train RMSE: 0.961314567555097
xgb_model Validation RMSE: 0.9790819270113794
svd_model Train RMSE: 0.9199768252693715
svd_model Validation RMSE: 1.006667691127854
Round 1
xgb_model Train RMSE: 0.9608258069687269
xgb_model Validation RMSE: 0.9844384057182686
svd_model Train RMSE: 0.9196969646306568
svd_model Validation RMSE: 1.0090727105211657
Round 2
xgb_model Train RMSE: 0.9612345896186997
xgb_model Validation RMSE: 0.9755859261259673
svd_model Train RMSE: 0.9205118879798375
svd_model Validation RMSE: 1.0024628152732387
Round 3
xgb_model Train RMSE: 0.9600560238073303
xgb_model Validation RMSE: 0.9850294464825607
svd_model Train RMSE: 0.9196324280716985
svd_model Validation RMSE: 1.0099941199662716
Round 4
xgb_model Train RMSE: 0.9609241614903443
xgb_model Validation RMSE: 0.9791185701674753
svd_model Train RMSE: 0.9201415267187989
svd_model Validation RMSE: 1.0055031797361567
Round 5
xgb_model Train RMSE: 0.9614772150961423
xgb_model Validation RMSE: 0.9771736761201681
svd_model Train RMSE: 0.9198196621731823
svd_model Validation RMSE: 1.0055345924057497
Round 6
xgb_model Train RMSE: 0.9608213217012966
xgb_model Validation RMSE: 0.9780857647512285
svd_model Train RMSE: 0.9209030796665477
svd_model Validation RMSE: 1.0047048481343381
Round 7
xgb_model Train RMSE: 0.9601332228937886
xgb_model Validation RMSE: 0.9871831216015755
svd_model Train RMSE: 0.9195282578874696
svd_model Validation RMSE: 1.012551982975392
Round 8
xgb_model Train RMSE: 0.9601667269734642
xgb_model Validation RMSE: 0.9882715165254962
svd_model Train RMSE: 0.9191144768604964
svd_model Validation RMSE: 1.0137609234107374
Round 9
xgb_model Train RMSE: 0.9599853542745727
xgb_model Validation RMSE: 0.9861736797748782
svd_model Train RMSE: 0.9190034756831778
svd_model Validation RMSE: 1.014038767504188
'''
print('step 11 finish')


# step 12: Select weight between svd and xgboost
reader = Reader(rating_scale=(1, 5))

svd_pred_r = []
xgb_pred_r = []

train_data = Dataset.load_from_df(train_df, reader)
trainset = train_data.build_full_trainset().build_testset()

for i in range(10):
    loaded_model = joblib.load(f'{model_folder}/model_{i}.joblib')
    svd_model = loaded_model['svd_model']
    svd_pred = svd_model.test(trainset)
    svd_pred_dict = {(prediction.uid, prediction.iid):prediction.est for prediction in svd_pred}
    svd_pred_r.append(svd_pred_dict)

    xgb_model = loaded_model['xgb_model']
    xgb_pred = xgb_model.predict(xtrain)
    xgb_pred_dict = {(line[0], line[1]): xgb_pred[i] for i, line in enumerate(train_rdd.collect())}
    xgb_pred_r.append(xgb_pred_dict)

for step in range(0,11):
    step_i = step/20
    agg_r = []
    for i, line in enumerate(train_rdd.collect()):
        svd_r = mean([svd_pred_r[model_i][(line[0], line[1])] for model_i in range(10)])
        xgb_r = mean([float(xgb_pred_r[model_i][(line[0], line[1])]) for model_i in range(10)])
        agg_r.append(svd_r*step_i+xgb_r*(1-step_i))
    rmse = sqrt(mean_squared_error(ytrain, agg_r))
    print(step_i,'RMSE', rmse)

print('step 12 finish')
