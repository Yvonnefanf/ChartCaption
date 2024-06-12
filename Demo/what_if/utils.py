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
    
    FEATURES=['sulphates', 'alcohol','total sulfur dioxide', 'volatile acidity','citric acid','free sulfur dioxide']
    
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
    org_importance = [abs(val) for val in exp_dict.values()]
    # Convert feature values to strings
    original_importance = {feature: abs(val) for feature, val in exp_dict.items()}

    instance_str = instance.apply(lambda x: str(x))
    original_instance_str = instance.apply(lambda x: str(x))

    # Extract feature names and values using regular expressions
    features_with_values = []
    features_ = []
    features_values = []
    
    org_importance_ = []
    
    for i,feature in enumerate(features):
        match = re.findall(r'([a-zA-Z]+)', feature)
        if match:
            feature_name = ' '.join(match)  # Combine all matches with a space
            if feature_name in FEATURES:
                
                feature_value = instance_str[feature_name]
                features_values.append(feature_value)
                features_with_values.append(f"{feature_name}={feature_value}")
                features_.append(f"{feature_name}")
                org_importance_.append(org_importance[i])
        else:
            features_with_values.append(f"{feature}={instance_str[feature]}")
    # Step 4: Plot Feature Importance with values
    fig, ax1= plt.subplots(figsize=(8, 5))
    ax1.set_yticklabels([])
    ax1.yaxis.set_ticks([])
    ax1.axvline(x=0, color='black',linewidth=0.6)
    bars = plt.barh(features_with_values, org_importance_, color='#20A6FF')
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
    gam_path = plot_gam_contributions(X_train, index,model)
    
    return image_path, features_with_values,features_values, features_, org_importance_,features_with_bound,gam_path



def plot_gam_contributions(X_train, index,model,instance_ = None,img_path_=''):

    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    prediction = model.predict(X_train)
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10)).fit(X_train, prediction)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    


    selected_features =[ 'volatile acidity', 'citric acid','free sulfur dioxide','total sulfur dioxide','sulphates', 'alcohol']
    selected_feature_indices = [X_train.columns.get_loc(feature) for feature in selected_features]

    instance = X_train.iloc[index]
    contributions = np.array([gam.partial_dependence(term=i, X=instance.values.reshape(1, -1)) for i in selected_feature_indices])
    if img_path_ != '':
        new_instance = instance_
        
        new_contributions = np.array([gam.partial_dependence(term=i, X=new_instance.values.reshape(1, -1)) for i in selected_feature_indices])


    for i, (feature, feature_idx) in enumerate(zip(selected_features, selected_feature_indices)):
        ax = axes[i]
        XX = gam.generate_X_grid(term=feature_idx)
        pdep, confi = gam.partial_dependence(term=feature_idx, width=0.9)
        
        ax.plot(XX[:, feature_idx], pdep)
        ax.fill_between(XX[:, feature_idx], confi[:, 0], confi[:, 1], alpha=0.1)
        
        # Highlight the instance's specific value on the shape function
        # ax.axvline(instance[i], color='#409EFF', linestyle='--')
        # ax.scatter(instance[i], contributions[i], color='#409EFF')

        # highlight
        color = 'orange' if contributions[i][0] >= 0 else '#01796f'
        # ax.axvline(instance[feature_idx], color=color, linestyle='--')
        ax.axhline(y=contributions[i][0], color=color, linestyle='--',linewidth=1)
        ax.scatter(instance[feature_idx], contributions[i][0], color=color)
        ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
        
        if img_path_ != '' and new_contributions[i][0] != contributions[i][0]:
            # ax.axvline(new_instance[feature_idx], color='deeppink', linestyle='--')
            ax.axhline(y=new_contributions[i][0], color='deeppink', linestyle='--',linewidth=1)
            ax.scatter(new_instance[feature_idx], new_contributions[i][0], color='deeppink')
        
        ax.set_title(f'{feature}',fontsize=18)
        # ax.set_xlabel(X_train.columns[i])
        # ax.set_ylabel('Partial Dependence')
        # Save the plot as an image file
    if img_path_ == '':
        image_path = 'imgs/gam_{}.png'.format(index)
    else:
        image_path = img_path_
    plt.tight_layout()
    plt.savefig(image_path,transparent=True)
    plt.close()
    return image_path
    
    





    