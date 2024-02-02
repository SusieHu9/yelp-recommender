from pyspark import SparkContext
import time
import json
import csv
import pandas as pd
import numpy as np
import joblib
import sys
from surprise import Dataset, Reader
from sklearn.metrics import mean_squared_error
from math import sqrt
from statistics import mean
from datetime import datetime

folder_path =  sys.argv[1] # ./data
test_file_path = sys.argv[2] # ./data/yelp_val.csv
output_file_path = sys.argv[3] # ./output.csv

sc = SparkContext('local[*]','Recommender')

start = time.time()
# step 1: Data Processing
# train_rdd = sc.textFile(folder_path + '/yelp_train.csv').filter(lambda line: line != 'user_id,business_id,stars')\
#     .map(lambda line: line.split(',')).cache()

test_rdd = sc.textFile(test_file_path)
test_first = test_rdd.first()
test_rdd = test_rdd.filter(lambda line: line != test_first)\
    .map(lambda line: line.split(',')).cache()
print('step 1 finish')


# step 2: Load result
model_folder = './model'
business_dict_base = joblib.load(f'{model_folder}/business_model_base.joblib')['business_model_base']
business_dummy_dict = joblib.load(f'{model_folder}/business_model_pca.joblib')['business_model_pca']
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

# step 3: Model-based CF - Create xtest, ytrue
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

xtest = np.array(test_rdd.map(lambda line: (line[0],line[1])).map(lambda line: extract_features(line)).collect(), dtype='float32')
# ytrue = np.array(test_rdd.map(lambda line: float(line[2])).collect(), dtype='float32')
print('step 3 finish')


# step 4: Model: SVD - Create testset
reader = Reader(rating_scale=(1, 5))
test_in_df = pd.DataFrame(test_rdd.map(lambda line: (line[0],line[1])).collect(),columns=['user_id', 'business_id'])
test_in_df['stars'] = 3.5
test_data = Dataset.load_from_df(test_in_df[['user_id', 'business_id', 'stars']], reader)
testset = test_data.build_full_trainset().build_testset()
print('step 4 finish')


# step 5: Predictions from Bagging
reader = Reader(rating_scale=(1, 5))
svd_pred_r = []
xgb_pred_r = []

for i in range(10):
    loaded_model = joblib.load(f'{model_folder}/model_{i}.joblib')
    svd_model = loaded_model['svd_model']
    svd_pred = svd_model.test(testset)
    svd_pred_dict = {(prediction.uid, prediction.iid):prediction.est for prediction in svd_pred}
    svd_pred_r.append(svd_pred_dict)

    xgb_model = loaded_model['xgb_model']
    xgb_pred = xgb_model.predict(xtest)
    xgb_pred_dict = {(line[0], line[1]): xgb_pred[i] for i, line in enumerate(test_rdd.collect())}
    xgb_pred_r.append(xgb_pred_dict)
print('step 5 finish')


# step 6: Get final prediction
agg_r = []
for i, line in enumerate(test_rdd.collect()):
    svd_r = mean([svd_pred_r[model_i][(line[0], line[1])] for model_i in range(10)])
    xgb_r = mean([float(xgb_pred_r[model_i][(line[0], line[1])]) for model_i in range(10)])
    agg_r_temp = svd_r*0.2+xgb_r*(1-0.2)
    if agg_r_temp > 5:
        agg_r.append(5)
    elif agg_r_temp < 1:
        agg_r.append(1)
    else:
        agg_r.append(agg_r_temp)
# rmse = sqrt(mean_squared_error(ytrue, agg_r))
# print('RMSE', rmse)
print('step 6 finish')


# step 7: Output
with open(output_file_path,'w',newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['user_id',' business_id',' prediction'])
    for i, line in enumerate(test_rdd.collect()):
        csv_writer.writerow([line[0], line[1], agg_r[i]])

# range_0 = 0
# range_1 = 0
# range_2 = 0
# range_3 = 0
# range_4 = 0
# abs_value = (np.array(ytrue) - np.array(agg_r)).tolist()
# for i in range(len(abs_value)):
#     if (abs_value[i] <1):
#         range_0 +=1
#     elif (abs_value[i] <2):
#         range_1 +=1
#     elif (abs_value[i] <3):
#         range_2 +=1
#     elif (abs_value[i] <4):
#         range_3 +=1
#     else:
#         range_4 +=1
# print('>=0 and <1:',range_0)
# print('>=1 and <2:',range_1)
# print('>=2 and <3:',range_2)
# print('>=3 and <4:',range_3)
# print('>=4:',range_4)
# print('step 7 finish')


end = time.time()
print('Duration:',end-start)
