# Import necessary libraries
import time  # To measure execution time
import pickle  # For loading the CatBoost model
import lightgbm as lgb  # LightGBM for boosting
import catboost as cb  # CatBoost for boosting
from tensorflow import keras  # Keras for loading CNN model
import numpy as np  # For numerical operations
import joblib  # To load models such as XGBoost and LightGBM
import tensorflow as tf  # TensorFlow for working with the CNN
from tensorflow.keras.models import load_model  # To load the saved CNN model
import pandas as pd  # For data manipulation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # To evaluate model performance
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler  # For scaling data

# Define paths to saved models
xgboost_model_path = "xgboost_model_565.pkl"
lightgbm_model_path = "lightgbm_model_565.pkl"
catboost_model_path = "catboost_model_565.pkl"
cnn_model_path = "cnn_checkpoint_565.h5"

print("Load model")
# Load the XGBoost model
xgboost_model = joblib.load(xgboost_model_path)
# Load the LightGBM model
lightgbm_model = joblib.load(lightgbm_model_path)
# Load the CatBoost model
with open(catboost_model_path, 'rb') as f:
    catboost_model = pickle.load(f)
# Load the CNN model
cnn_model = load_model(cnn_model_path)

# File paths for training and test data
train_file_path = "X_train.dat"
test_file_path = "X_test.dat"
y_train_file_path = "y_train.dat"
y_test_file_path = "y_test.dat"
# Size of the dataset
train_size = 800000
test_size = 200000
columns = 2381

# Function to load data using memory-mapped files
def load_data(file_path, size, y_file_path=None):
    data = np.memmap(file_path, dtype=np.float32, mode="r", shape=(size, columns))
    if y_file_path:
        labels = np.memmap(y_file_path, dtype=np.float32, mode="r", shape=size)
        rows = (labels != -1)
        data = data[rows]
        labels = labels[rows]
        return data, labels
    return data

print("Load training set")
X_train = load_data(train_file_path, train_size, y_train_file_path)
print("Load testing set")
X_test = load_data(test_file_path, test_size, y_test_file_path)

# Convert data into pandas DataFrames
X_train_df = pd.DataFrame(X_train[0])
y_train_df = pd.Series(X_train[1])
X_test_df = pd.DataFrame(X_test[0])
y_test_df = pd.Series(X_test[1])

# List of selected features (feature intersection from different models)
selected_features_intersection = [691, 801, 930, 605, 2359, 692, 152, 786, 513, 637, 833, 2353, 579, 130, 128, 2364, 765, 615, 620, 492, 105, 2354, 617, 129, 522, 1608, 736, 44, 48, 88, 373, 2360,
                                  49, 603, 563, 921, 108, 784, 527, 359, 390, 531, 566, 419, 2355, 388, 232, 2375, 168, 533, 120, 109, 554, 64, 2362, 266, 616, 488, 611, 589, 140, 588, 550, 655, 586,
                                  613, 256, 2351, 254, 577, 584, 38, 89, 498, 222, 587, 734, 1043, 771, 547, 555, 185, 196, 496, 689, 825, 1941, 686, 20, 501, 255, 422, 607, 2356, 609, 733, 224, 230,
                                  490, 583, 551, 994, 29, 164, 248, 153, 124, 570, 480, 104, 392, 386, 688, 3, 234, 193, 1060, 596, 160, 369, 503, 626, 598, 797, 273, 2361, 51, 2363, 511, 149, 208, 574,
                                  683, 365, 4, 125, 146, 383, 514, 410, 581, 233, 539, 312, 530, 315, 50, 47, 144, 2352, 592, 420, 618, 33, 836, 658, 552, 200, 9, 414, 133, 677, 573, 347, 557, 371, 1561,
                                  1559, 41, 447, 402, 319, 19, 807, 640, 304, 523, 827, 2376, 68, 542, 707, 55, 484, 2020, 227, 540, 7, 1650, 99, 595, 431, 23, 385, 418, 535, 451, 487, 405, 0, 314, 360,
                                  597, 114, 406, 1, 271, 186, 131, 565, 112, 489, 1174, 80, 578, 245, 413, 599, 458, 437, 680, 463, 92, 519, 240, 210, 334, 236, 582, 34, 5, 187, 95, 975, 85, 835, 632, 593,
                                  493, 12, 351, 344, 569, 559, 30, 524, 834, 486, 432, 601, 378, 571, 415, 785, 116, 1011, 398, 545, 986, 290, 212, 156, 40, 228, 172, 122, 561, 376, 123, 580, 1431, 387, 454,
                                  549, 332, 973, 532, 86, 2159, 2004, 299, 538, 1511, 94, 70, 558, 189, 393, 364, 548, 357, 77, 534, 52, 8, 362, 1160, 75, 132, 353, 510, 567, 253, 249, 509, 591, 25, 389, 1566,
                                  113, 372, 335, 506, 165, 37, 2093, 6, 226, 516, 345, 543, 90, 528, 340, 602, 475, 106, 1597, 560, 21, 472, 1443, 213, 564, 15, 250, 176, 110, 1201, 119, 98, 192, 1836, 2378, 225,
                                  721, 681, 546, 757, 778, 515, 529, 407, 229, 117, 221, 585, 195, 350, 408, 1266, 2373, 612, 2078, 247, 590, 1190, 239, 685, 497, 367, 622, 575, 526, 65, 682, 337, 1569, 36, 141,
                                  143, 374, 126, 494, 639, 1526, 1241, 517, 137, 97, 246, 375, 409, 171, 536, 2380, 111, 298, 63, 1697, 544, 1249, 537, 384, 424, 1689, 81, 2117, 698, 610, 1298, 608, 1377, 1799,
                                  810, 84, 159, 2162, 809, 562, 1733, 1853, 457, 74, 1812, 67, 323, 2372, 1404, 798, 251, 423, 24, 1451, 1756, 1750, 416, 556, 553, 175, 1209, 145, 600, 241, 401, 231, 572, 2095,
                                  1286, 43, 204, 76, 121, 430, 568, 150, 665, 2009, 412, 1927, 377, 1366, 1144, 508, 56, 1786, 71, 166, 115, 203, 1246, 391, 399, 1546, 60, 426, 134, 594, 417, 2012, 504, 102, 198,
                                  342, 1645, 2371, 32, 951, 69, 541, 502, 13, 619, 461, 243, 440, 660, 775, 223, 158, 127, 710, 604, 411, 211, 138, 59, 748, 507, 139, 42, 512, 715, 53, 1553, 289, 2210, 199, 576,
                                  11, 642, 87, 35, 525, 476, 499, 147, 1835, 434, 107, 1780, 16, 287, 460, 61, 194, 181, 968, 2, 679, 66, 403, 623, 2028, 495, 191, 183, 482, 452, 500, 27, 244, 100, 83, 438, 62,
                                  103, 725, 142, 10, 22, 78, 31, 654, 219]

# Only select the features from the intersection
X_train_selected = X_train_df[list(selected_features_intersection)]
X_test_selected = X_test_df[list(selected_features_intersection)]
print(len(selected_features_intersection))

columns = 565
# Initialize the scaler
scaler = StandardScaler()

# Scale the training data
X_train_cnn = scaler.fit_transform(X_train_selected)

# Scale the test data
X_test_cnn = scaler.transform(X_test_selected)

# Reshape the data to fit the input of the CNN model
X_train_cnn = X_train_cnn.reshape(-1, columns, 1)
X_test_cnn = X_test_cnn.reshape(-1, columns, 1)

# Predictions on the test set
catboost_pred_test = catboost_model.predict_proba(X_test_selected)[:, 1]
xgboost_pred_test = xgboost_model.predict_proba(X_test_selected)[:, 1]
lightgbm_pred_test = lightgbm_model.predict_proba(X_test_selected)[:, 1]
cnn_pred_test = cnn_model.predict(X_test_cnn).flatten()  # CNN predictions on the test set

# Perform soft voting (average the predictions of the models)
start_time_voting = time.time()
soft_voting_pred_test = (catboost_pred_test + xgboost_pred_test + lightgbm_pred_test + cnn_pred_test) / 4
end_time_voting = time.time()
voting_time = start_time_voting - end_time_voting
print(f"Voting time: {voting_time:.5f} seconds")

# Predictions on the training set
catboost_pred_train = catboost_model.predict_proba(X_train_selected)[:, 1]
xgboost_pred_train = xgboost_model.predict_proba(X_train_selected)[:, 1]
lightgbm_pred_train = lightgbm_model.predict_proba(X_train_selected)[:, 1]
cnn_pred_train = cnn_model.predict(X_train_cnn).flatten()  # CNN predictions on the training set
soft_voting_pred_train = (catboost_pred_train + xgboost_pred_train + lightgbm_pred_train + cnn_pred_train) / 4

# Create DataFrame with predictions from base models for the test set
base_model_predictions_test = pd.DataFrame({
    'CatBoost': catboost_pred_test,
    'XGBoost': xgboost_pred_test,
    'LightGBM': lightgbm_pred_test,
    'CNN': cnn_pred_test,
    'SoftVoting': soft_voting_pred_test
})

# Create DataFrame with predictions from base models for the training set
base_model_predictions_train = pd.DataFrame({
    'CatBoost': catboost_pred_train,
    'XGBoost': xgboost_pred_train,
    'LightGBM': lightgbm_pred_train,
    'CNN': cnn_pred_train,
    'SoftVoting': soft_voting_pred_train
})

# Meta-model training using XGBoost
import xgboost as xgb
xgb_params = {
    'n_estimators': 15000,
    'learning_rate': 0.7,
    'objective': 'binary:logistic',  # Binary classification
    'tree_method': 'gpu_hist',  # Use GPU for training
    'n_jobs': -1,  # Number of threads to use
    'eval_metric': 'logloss'  # Evaluation metric
}
meta_model = xgb.XGBClassifier(**xgb_params)
meta_model.fit(base_model_predictions_train, y_train_df)

# Meta-model predictions on the test set
start_time_stacking = time.time()
meta_pred_proba = meta_model.predict_proba(base_model_predictions_test)[:, 1]
end_time_stacking = time.time()
stacking_time = start_time_stacking - end_time_stacking
print(f"stacking_time: {stacking_time:.5f} seconds")
meta_pred_binary = (meta_pred_proba > 0.5).astype(int)

# Parallel predictions for CNN, XGBoost, CatBoost, and LightGBM using concurrent futures
import concurrent.futures

def predict(model_type):
    result = []
    preds = None

    if model_type == "CNN":
        print("CNN Prediction")
        elapsed = float('inf')
        for i in range(LOOP):
            start = time.time()
            preds = cnn_model.predict(X_test_cnn).flatten()  # CNN predictions
            t = time.time() - start
            if t < elapsed:
                elapsed = t
        result.append(elapsed)
        result.append(preds)

    elif model_type == "XGB":
        print("XGB Prediction")
        elapsed = float('inf')
        for i in range(LOOP):
            start = time.time()
            preds = xgboost_model.predict_proba(X_test_selected)[:, 1]  # XGBoost predictions
            t = time.time() - start
            if t < elapsed:
                elapsed = t
        result.append(elapsed)
        result.append(preds)

    elif model_type == "CatBoost":
        print("CatBoost Prediction")
        elapsed = float('inf')
        for i in range(LOOP):
            start = time.time()
            preds = catboost_model.predict_proba(X_test_selected)[:, 1]  # CatBoost predictions
            t = time.time() - start
            if t < elapsed:
                elapsed = t
        result.append(elapsed)
        result.append(preds)

    elif model_type == "LightGBM":
        print("LightGBM Prediction")
        elapsed = float('inf')
        for i in range(LOOP):
            start = time.time()
            preds = lightgbm_model.predict_proba(X_test_selected)[:, 1]  # LightGBM predictions
            t = time.time() - start
            if t < elapsed:
                elapsed = t
        result.append(elapsed)
        result.append(preds)

    return result

# Run parallel predictions
def run_parallel_predictions():
    print("Parallel Predicting...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        start_parallel = time.time()
        futures = {executor.submit(predict, model): model for model in ["CNN", "XGB", "CatBoost", "LightGBM"]}
        ret = {}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                ret[name] = future.result()
            except Exception as e:
                print('%r generated an exception: %s' % (name, e))
        parallel_time = time.time() - start_parallel
        print(f"parallel_time: {parallel_time:.5f} seconds")
        return ret

LOOP = 20
parallel_predictions = run_parallel_predictions()

# Evaluate accuracy and other metrics
def evaluate_performance(pred_binary, true_labels):
    acc = accuracy_score(true_labels, pred_binary)
    precision = precision_score(true_labels, pred_binary)
    recall = recall_score(true_labels, pred_binary)
    f1 = f1_score(true_labels, pred_binary)
    roc_auc = roc_auc_score(true_labels, pred_binary)
    return acc, precision, recall, f1, roc_auc

print("Metrics:")
acc, precision, recall, f1, roc_auc = evaluate_performance(meta_pred_binary, y_test_df)
print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
