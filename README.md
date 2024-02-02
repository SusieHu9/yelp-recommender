# README
# A Hybrid Recommender with Yelp Dataset Challenge

## Introduction
This project aims to build a recommendation system for Yelp to predict the ratings that users will give to businesses. In this project, I implemented a hybrid recommendation system within a PySpark distributed environment. I employed `a weighted hybrid approach`, which combines two model-based hybrid components through a linear function. The first component uses a matrix factorization-based approach, employing `Singular Value Decomposition (SVD)`. Hyperparameter tuning for SVD was conducted using GridSearchCV from the surprise package. The second component involves a model-based approach, utilizing `XGBRegressor` (a regressor based on Decision Trees). Feature extraction and hyperparameter tuning were performed, with the latter using GridSearchCV from the sklearn package.

### Why is it original?
- Boosted model efficiency by conducting exploratory data analysis on a 3GB JSON dataset with PySpark RDD, applying dimension reduction techniques, `Principal Component Analysis (PCA)` to reduce 3000 dummy variables to 30 features, conserving 5GB system RAM and reducing 50% runtime

- Augmented model accuracy through feature extraction, hyperparameter tuning, a combination of `Bagging and Stacking` in `ensemble learning` (trained 10 SVD models and 10 XGBRegressor models, weighted equally each kind of models' prediction), and weight optimization between SVD and XGBRegressor, resulting in a 24% reduction in test RMSE compared to the baseline model

### Model Results:
The model's accuracy and efficiency on the validation dataset are displayed below.
```
Error Distribution:
>=0 and <1: 123548
>=1 and <2: 17670
>=2 and <3: 794
>=3 and <4: 32
>=4: 0

RMSE:
0.9771748984859657

Execution Time:
192.3451018333435
```
On the hidden test dataset, the RMSE result reaches approximately 0.976.

## Getting Started
### Prerequisite:
```
pip install scikit-surprise
pip install pandas
pip install numpy
pip install scikit-learn
pip install xgboost
```

### Programming Environment
```
Python 3.6, JDK 1.8, and Spark 3.1.2
```

### Run the recommender:
To run the Python code, follow these steps:
1. Make sure your current programming environment meets the requirements
2. Clone the repository to your local machine
3. Download the dataset used by the recommender from [Google Drive](https://drive.google.com/drive/folders/1AbJWyM3_bcbdsZFlO0ppTgVT3p9YQvMs?usp=sharing). Save these files in the repo's `data` folder.
4. Use Spark to execute the file (adjusting the PySpark location as necessary):
    ```
    /spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G recommender.py ./data ./data/yelp_val.csv ./output.csv
    ```
5. The prediction result will be saved in `./output.csv`

## Project Development
### Data Source: 
* [Yelp dataset](https://www.yelp.com/dataset)

### Data Description:
* yelp_train.csv: the training data, which only includes the columns: user_id, business_id, and stars.
* yelp_val.csv: the validation data, which are in the same format as training data.
* review_train.json: review data only for the training pairs (user, business)
* user.json: all user metadata
* business.json: all business metadata, including locations, attributes, and categories
* checkin.json: user check-ins for individual businesses
* tip.json: tips (short reviews) written by a user about a business
* photo.json: photo data, including captions and classifications

### File Description:
* recommender.py: recommender codes
* train.py: code for training models
* model: models trained by train.py and required in recommender.py
* data: dataset folder
