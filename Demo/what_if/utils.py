import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f
import numpy as np
import re
from matplotlib.collections import LineCollection

import random
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

def get_model():
    np.random.seed(random_seed)
    random.seed(random_seed)
    # Step 1: Load the Dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, delimiter=";")

    # Step 2: Train a Model
    X = data.drop("quality", axis=1)
    y = data['quality'].apply(lambda x: 1 if x >= 7 else 0)  # 2 class: >=7 high quality <7 low quality

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=random_seed)
    model.fit(X_train, y_train)
    y_test = np.array(y_test)
    pred = model.predict(X_train)
    y_train= np.array(y_train)
    
    k = 0
    n=0
    for i in range(len(pred)):
        if pred[i] == y_train[i]:
            k+=1
            if y_train[i]==1:
                # print(i)
                n+=1
    print('acc:',k/len(pred))
    print('high quality:',n/len(pred))
    return model, X_train, y_train,X,y


def org_instance_importance(model, X_train, y_train,X,index):
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    index = int(index)
    print("index",index)
    
    instance = X_train.iloc[index]
    # Generate explanation
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, 
                                                   feature_names=X.columns, 
                                                   class_names=['<7', '>=7'], 
                                                   discretize_continuous=True,
                                                   random_state=random_seed)  # Set the random state here)
    exp = explainer.explain_instance(instance.values, model.predict_proba, num_features=len(X.columns))
    # Extract feature importance (absolute values)
    exp_dict = dict(exp.as_list())
    features = list(exp_dict.keys())
    importance = [abs(val) for val in exp_dict.values()]
    org_importance = [abs(val) for val in exp_dict.values()]
    # Convert feature values to strings
    original_importance = {feature: abs(val) for feature, val in exp_dict.items()}

    instance_str = instance.apply(lambda x: str(x))
    original_instance_str = instance.apply(lambda x: str(x))

    # Extract feature names and values using regular expressions
    features_with_values = []
    features_ = []
    features_values = []
    max_importance = max(importance)
    for feature in features:
        match = re.findall(r'([a-zA-Z]+)', feature)
        if match:
            feature_name = ' '.join(match)  # Combine all matches with a space
            feature_value = instance_str[feature_name]
            features_values.append(feature_value)
            features_with_values.append(f"{feature_name}={feature_value}")
            features_.append(f"{feature_name}")
        else:
            features_with_values.append(f"{feature}={instance_str[feature]}")
    # Step 4: Plot Feature Importance with values
    fig, ax1= plt.subplots(figsize=(5, 5))
    ax1.set_yticklabels([])
    ax1.yaxis.set_ticks([])
    ax1.axvline(x=0, color='black',linewidth=0.6)
    bars = plt.barh(features_with_values, org_importance, color='#20A6FF')
    plt.xlabel('Feature Importance')
    # plt.title('LIME Feature Importance for Instance {}'.format(index))
    plt.gca().invert_yaxis()
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.subplots_adjust(left=0)  # Adjust the left margin to fit long labels
    # Custom y-axis labels with min, max, and current values
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)
    features_with_bound =[]
    org_features=dict()
    for i, bar in enumerate(bars):
        feature = features_with_values[i]
        feature_name = feature.split('=')[0]
        min_val = X_train[feature_name].min()
        max_val = X_train[feature_name].max()
        current_val = instance[feature_name]
        features_with_bound.append({'name':feature_name,'range':[min_val,current_val,max_val] })
        org_features[feature_name] = current_val
        # Normalize values to a fixed range (e.g., 0 to 1)
        
    # Save the plot as an image file
    image_path = 'imgs/lime_feature_importance_instance_{}.png'.format(index)
    plt.savefig(image_path,transparent=True)
    plt.close()
    return image_path, features_with_values,features_values, features_, importance,features_with_bound




    
    





    