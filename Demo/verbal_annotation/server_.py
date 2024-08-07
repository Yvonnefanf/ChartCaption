from flask import Flask, request, jsonify,send_from_directory
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS, cross_origin
import json
import numpy as np
import random
import lime
import re
from sklearn.metrics.pairwise import euclidean_distances
from datetime import datetime
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import shap

random_seed = 42
import matplotlib.pyplot as plt
np.random.seed(random_seed)
random.seed(random_seed)
import joblib
import pickle

df = pd.read_csv('./data/secondary_data.csv',delimiter=';')

labelencoder = LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])


y = df['class']

# columns_to_drop = ['class','cap-surface', 'gill-attachment', 'gill-spacing', 'stem-root','stem-surface','veil-type','veil-color','ring-type','spore-print-color']  # 示例列名，请根据实际情况替换
columns_to_drop = ['class','cap-shape','cap-color','gill-color','cap-surface','stem-color', 'gill-attachment', 'gill-spacing', 'stem-root','stem-surface','veil-type','veil-color','ring-type','spore-print-color','habitat']  # 示例列名，请根据实际情况替换

# 删除多个列
X = df.drop(columns=columns_to_drop)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# load mdoel
model = joblib.load('./data/xgb_model.joblib')


MAX_ABS_SHAP_VALUE = 8.049500501404982


#Step 3: Apply LIME to Explain a Single Instance
features_with_values = []
org_importance = []
features_ = []
features_values = []
app = Flask(__name__,static_folder='')
cors = CORS(app, supports_credentials=True)

@app.route("/", methods=["GET", "POST"])
def index():
    # return render_template("SilasIndex.html")
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/test", methods=["GET", "POST"])
def test_interface():
    # return render_template("SilasIndex.html")
    return send_from_directory(app.static_folder, 'pdp_recall.html')


def is_continuous(feature):
    return X_train[feature].nunique() > 20  # 假设唯一值超过20个认为是连续特征


@app.route('/get_instance_data_XAI', methods=['POST','GET'])
@cross_origin()
def caption_gen():
    global features_with_values, org_importance, features_,features_values, X_train,model
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    data = request.get_json()
    index = data.get('index')
    index = int(index)
    print("index111",index)
    instance = X_train.iloc[index]
    prediction = model.predict([instance])
    print("prediction",prediction[0])
    
    # FEATURES =['cap-diameter', 'cap-shape', 'cap-color', 'does-bruise-or-bleed', 'gill-color', 'stem-height', 'stem-width', 'stem-color', 'has-ring', 'habitat', 'season']
    FEATURES =['cap-diameter', 'does-bruise-or-bleed', 'stem-height', 'stem-width', 'has-ring', 'season']
    # FEATURES=['sulphates', 'alcohol','total sulfur dioxide', 'volatile acidity','citric acid','free sulfur dioxide']
    features_ = []
    features_with_bound = []
    features_values=[]
    # shap_value = shap_values[index]
    explainer = shap.Explainer(model, X_train)
    shap_value = explainer(instance)
    # Create the order_ array based on sorted importance
    order_ = [0] * len(FEATURES)
   
    for i in range(len(FEATURES)):
        feature = FEATURES[i]
        importance = float(shap_value.values[i])
        normalized_importance = 100 / MAX_ABS_SHAP_VALUE * importance
        categorical = False
        if is_continuous(feature):
            min_val = float(X_train[feature].min())
            max_val = float(X_train[feature].max())
            val = float(instance[feature])
            
        else:
            unique_values = X_train[feature].unique()
            min_val = 0
            max_val = len(unique_values)
            categorical=True
            val=float(X_train[feature].values[index])+1
            
            
            
        
        features_.append(feature)
        features_values.append(val)
        order = order_[i]
        # if feature == 'free sulfur dioxide':
        #     feature = 'SO2'
        # # if feature == 'total sulfur dioxide':
        # #     feature = 'total SO2'
        # if feature == 'volatile acidity':
        #     feature = 'vinegar-taint'
        # if feature == 'fixed acidity':
        #     feature = 'fixed-acidity'
        # if feature == 'residual sugar':
        #     feature = 'sweetness'
        
        features_with_bound.append({'name': feature, 'range':[min_val,val,max_val],'importance':normalized_importance,'importance_order':order,'categorical':categorical })
        
        # Sort features by the absolute value of their importance
    
            
    # img_path,features_with_values,features_values, features_, org_importance,features_with_bound,gam_path = org_instance_importance(model, X_train, y_train,X, index)
    print("features_with_bound",features_with_bound)
    return jsonify({'features_with_bound':features_with_bound,"value_list":features_values,"predict":int(prediction[0])})

def getLIMEAttribution(i):
    global features_with_values, org_importance, features_,features_values, X_train,model
    X_instance = X_train.iloc[i]
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, 
                                                   feature_names=X.columns, 
                                                   class_names=['<7', '>=7'], 
                                                   discretize_continuous=True)

    # Generate explanations for a specific instance (replace X_instance with your instance)
    exp = explainer.explain_instance(X_instance, model.predict_proba, num_features=len(X_train.columns))

    # Get feature weights as a dictionary
    feature_weights = exp.local_exp[1]  # Use [0] for the '<7' class or [1] for '>=7' class

    # Normalize to -100 to +100 scale
    max_abs_weight = max(abs(weight) for feature, weight in feature_weights)
    normalized_importance = {feature: weight * 100 / max_abs_weight for feature, weight in feature_weights}

    # Create final dictionary in the desired format
    final_importance = {X_train.columns[idx]: normalized_importance.get(idx, 0) for idx in range(len(X_train.columns))}

    return  final_importance

@app.route('/update_annotation', methods=['POST','GET'])
@cross_origin()
def update_annotation():
    # FEATURES=['sulphates', 'alcohol','total sulfur dioxide', 'volatile acidity','citric acid','free sulfur dioxide']
    FEATURES = ['fixed acidity',	'volatile acidity',	'citric acid',	'residual sugar',	'chlorides',	'free sulfur dioxide',	'total sulfur dioxide',	'density','pH','sulphates','alcohol']
    global features_with_values, org_importance, features_
    random_seed = 42

    np.random.seed(random_seed)
    random.seed(random_seed)
    
    data = request.get_json()
    instance_index = data.get('index')
    new_value_list = data.get('value_list')
    # _,features_with_values,features_values, features_, org_importance,features_with_bound = org_instance_importance(model, X_train, y_train,X,explainer, instance_index)
    instance_index = int(instance_index)
    modified_features=[]
    update_values=[]
    for i in range(len(new_value_list)):
        print('hhhh',features_values[i],new_value_list[i],abs(float(features_values[i]) - float(new_value_list[i])))
        if abs(float(features_values[i]) - float(new_value_list[i]))>1e-3:
            modified_features.append(features_[i])
            update_values.append(new_value_list[i])
    # print(X_train.iloc[instance_index])
    # instance = X_train.iloc[instance_index].copy() 
    instance = X_train.iloc[instance_index].copy()
    
    for i in range(len(modified_features)):
        modified_key = modified_features[i]
        instance[modified_key] = update_values[i]
    print("org_importance",org_importance)
    predict = model.predict([instance])
    print("predict",predict[0])
    
    return jsonify({"predict":predict[0]})



@app.route('/get_similar_instance', methods=['POST','GET'])
@cross_origin()
def get_similar():
    data = request.get_json()
    target_feature = data.get('feature_name')
    instance_index = data.get('index')
    instance_index=int(instance_index)
    
    features = X_train.drop(target_feature, axis=1)
    
    # Calculate pairwise Euclidean distances between all instances
    distances = euclidean_distances(features, features)
    # Create a DataFrame from the distance matrix, setting infinite distance for diagonal (self-pair)
    distance_df = pd.DataFrame(distances, index=X_train.index, columns=X_train.index)
    np.fill_diagonal(distance_df.values, np.inf)
    
    # Calculate the absolute differences in the target feature for all pairs
    target_diffs = abs(X_train[target_feature].values[:, None] - X_train[target_feature].values)
    
    # Create a DataFrame from the target differences
    target_diff_df = pd.DataFrame(target_diffs, index=X_train.index, columns=X_train.index)

    # Find the pair with the minimum distance and maximum target feature difference
    # Use a combined metric or prioritize one over the other
    score_df = distance_df + (1 - (target_diff_df / target_diff_df.max().max()))
    
    # Find the index of the minimum score
    min_score_idx = score_df.stack().idxmin()
    most_similar, most_different = min_score_idx

    print(f"Most similar pair with highest difference in {target_feature}: Index {X_train.index.get_loc(most_similar)} and Index {X_train.index.get_loc(most_different)}")
    print(f"Difference in {target_feature}: {target_diff_df.loc[most_similar, most_different]}")
    print(f"Distance between instances: {distance_df.loc[most_similar, most_different]}")


    
    return jsonify({'ab':1})
    
    
    
@app.route('/instance_retrieval', methods=['POST','GET'])
@cross_origin()
def instance_retrieval():
    random_seed = 42

    np.random.seed(random_seed)
    random.seed(random_seed)
    
    data = request.get_json()
    lessThan = float(data.get('lessThan'))
    moreThan = float(data.get('moreThan'))
    instance_index = data.get('index')
    instance_index = int(instance_index)
    retrieve_instance_type = data.get('type')
    predictions = model.predict(X_train)
    # Get the given instance
    instance = X_train.iloc[instance_index].copy()
    # Separate instances by prediction type
    instances_0 = X_train[predictions <= lessThan]
    instances_1 = X_train[predictions >= moreThan ]
    print(len(instances_0),len(instances_1))
    
    if retrieve_instance_type == 'similar':
        ins_0, ins_1 = similar_retrival(instance, instances_0,instances_1)
    elif retrieve_instance_type == 'prototype':
        ins_0, ins_1 = prototype_retrival(instance, instances_0,instances_1)
    print("this step 3")
    # Convert to list of dictionaries for JSON response
    def convert_to_feature_dict(instance):
        # return {f'feature_{i}': value for i, value in enumerate(instance)}
        return instance.to_dict()
    
    response = {
        'instance_0': convert_to_feature_dict(ins_0),
        'instance_1': convert_to_feature_dict(ins_1)
    }
    return jsonify(response)

def similar_retrival(instance, instances_0,instances_1):
    # Calculate nearest neighbors within the same prediction class
    def find_nearest_neighbors(instances, instance, n_neighbors=1):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(instances)
        distances, indices = nbrs.kneighbors([instance])
        return indices[0][1:]  # Exclude the first index (itself)
    
    # Find nearest neighbors for the given instance in class 0 and class 1
    nearest_0_index = find_nearest_neighbors(instances_0, instance)
    nearest_1_index = find_nearest_neighbors(instances_1, instance)
    print("this step 2")
    # Retrieve the most similar instances
    most_similar_0 = instances_0.iloc[nearest_0_index[0]]
    most_similar_1 = instances_1.iloc[nearest_1_index[0]]
    return most_similar_0, most_similar_1

def prototype_retrival(instance, instances_0,instances_1):
    general_case_0 = instances_0.median()  
    general_case_1 = instances_1.median()  
    return general_case_0,general_case_1
  
  
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',port=6006)
