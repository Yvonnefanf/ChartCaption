from flask import Flask, request, jsonify,send_from_directory
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS, cross_origin
import json
from utils import org_instance_importance,get_model,plot_gam_contributions
import numpy as np
import random
import lime
import re
from datetime import datetime
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

random_seed = 42
import matplotlib.pyplot as plt
np.random.seed(random_seed)
random.seed(random_seed)

model, X_train, y_train,X,y = get_model()



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

@app.route("/chain", methods=["GET", "POST"])
def chain():
    # return render_template("SilasIndex.html")
    return send_from_directory(app.static_folder, 'index_chain.html')

@app.route("/feature_importance", methods=["GET", "POST"])
def feature_importance():
    # return render_template("SilasIndex.html")
    return send_from_directory(app.static_folder, 'feature_importance.html')


@app.route('/get_org_instance_chart', methods=['POST','GET'])
@cross_origin()
def caption_gen():
    global features_with_values, org_importance, features_,features_values
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    data = request.get_json()
    index = data.get('index')
    print("index111",index) 

    img_path,features_with_values,features_values, features_, org_importance,features_with_bound,gam_path = org_instance_importance(model, X_train, y_train,X, index)
   
    print("features_values",features_values)
    return jsonify({'img_path': img_path,'features_with_bound':features_with_bound,"value_list":features_values,'gam_path':gam_path})
  
@app.route('/update_annotation', methods=['POST','GET'])
@cross_origin()
def update_annotation():
    FEATURES=['sulphates', 'alcohol','total sulfur dioxide', 'volatile acidity','citric acid','free sulfur dioxide']
    
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
    print("predict",int(predict[0]))
    np.random.seed(random_seed)  # Ensure the random seed is set again before calling explain_instance
    random.seed(random_seed)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, 
                                                   feature_names=X.columns, 
                                                   class_names=['<7', '>=7'], 
                                                   discretize_continuous=True,
                                                   random_state=random_seed)  # Set the random state here)
    exp = explainer.explain_instance(instance.values, model.predict_proba, num_features=len(X.columns))

    # Extract feature importance (absolute values)
    exp_dict = dict(exp.as_list())
    features = list(exp_dict.keys())
    new_importance_ = [abs(val) for val in exp_dict.values()]
    
  
    instance_str = instance.apply(lambda x: str(x))

    # Extract feature names and values using regular expressions
    modified_features_with_values = []
    modified_features = []
    new_importance = []
    for i,feature in enumerate(features):
        match = re.findall(r'([a-zA-Z]+)', feature)
        # print("match",feature, match)
        if match:
            feature_name = ' '.join(match)  # Combine all matches with a space
            if feature_name in FEATURES:
                feature_value = instance_str[feature_name]
                modified_features_with_values.append(f"{feature_name}={feature_value}")
                modified_features.append(f"{feature_name}")
                new_importance.append(new_importance_[i])
        else:
            modified_features_with_values.append(f"{feature}={instance_str[feature]}")
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)

        # Create a mapping from modified feature names to their new importance
    new_importance_dict = dict(zip(modified_features, new_importance))
    print("new_importance_dict",new_importance_dict,feature,org_importance)

    # Plot Feature Importance with values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_yticklabels([])
    ax.yaxis.set_ticks([])
    prediction = int(predict[0])
    print("prediction",prediction)
    if prediction > 0:
        barColor = 'orange'
    else:
        barColor = '#01796f'
    
    
    bars = ax.barh(features_with_values, org_importance, color=barColor,alpha=0.4, label='Original Importance')

    # Add a black vertical line at x=0
    plt.axvline(x=0, color='black', linewidth=1)

    # Annotate bars with the new importance values and changes
    for i, (bar, org_value, feature) in enumerate(zip(bars, org_importance, features_)):
        bar_center = bar.get_y() + bar.get_height() / 2
        new_value = new_importance_dict[feature]
        if abs(new_value-org_value)>0.001:
            # print(abs(new_value-org_value))
            ax.plot([new_value], [bar_center], 'o', color='deeppink')  # Plot the new importance as an orange circle
            ax.vlines(new_value, bar_center - bar.get_height() / 2, bar_center + bar.get_height() / 2, colors='deeppink', linestyles='dashed')  # Add a vertical dashed line
            ax.annotate(f"Diff: {new_value-org_value:.3f}", (new_value, bar_center), textcoords="offset points", xytext=(5, 5), ha='left', color='deeppink')

    # # Add feature values next to the y-axis and on the bars
    # for idx, feature in enumerate(features_with_values,):

    #     if modified_key in feature:
    #         ax.text(-0.01, idx + 0.3, f"{modified_key}={modified_value:.2f}", va='center', ha='right', color='deeppink', fontsize=10, transform=ax.get_yaxis_transform())

    plt.xlabel('Feature Importance')
    # plt.title('Modified Instances with Annotations')


    custom_lines = [Line2D([0], [0], color='skyblue', lw=4, alpha=0.6),
                    Line2D([0], [0], color='deeppink', lw=2, linestyle='--')]
    ax.legend(custom_lines, ['Original Importance', 'Annotation for Change'], loc='lower right')

    # plt.legend(loc='lower right')
    plt.gca().invert_yaxis()
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.subplots_adjust(left=0)  #
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Save the plot as an image file
    # Convert modified_features and update_values to string format
    modified_features_str = '_'.join(modified_features)
    update_values_str = '_'.join(map(str, update_values))

    # Combine them into the file name
    image_path = 'imgs/lime_feature_importance_newinstance_{}_{}_{}.png'.format(instance_index, modified_features_str, update_values_str)

    plt.savefig(image_path,transparent=True)
    plt.close()
    return jsonify({'img_path': image_path,'features_with_bound':[],"value_list":[],"predict":int(predict[0])})

@app.route('/update_gam_annotation', methods=['POST','GET'])
@cross_origin()
def update_gam_annotation():

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
    print("predict",int(predict[0]))
    
    modified_features_str = '_'.join(modified_features)
    update_values_str = '_'.join(map(str, update_values))
    
    image_path = 'imgs/gam_newinstance_{}_{}_{}.png'.format(instance_index, modified_features_str, update_values_str)
    _ = plot_gam_contributions(X_train,instance_index,model,instance,image_path)
    return jsonify({'img_path': image_path,'predict':int(predict[0])})
    
    
    
    
    
@app.route('/instance_retrieval', methods=['POST','GET'])
@cross_origin()
def instance_retrieval():
    random_seed = 42

    np.random.seed(random_seed)
    random.seed(random_seed)
    
    data = request.get_json()
    instance_index = data.get('index')
    instance_index = int(instance_index)
    retrieve_instance_type = data.get('type')
    predictions = model.predict(X_train)
    # Get the given instance
    instance = X_train.iloc[instance_index].copy()
    # Separate instances by prediction type
    instances_0 = X_train[predictions == 0]
    instances_1 = X_train[predictions == 1]
    
    if retrieve_instance_type == 'similar':
        most_similar_0, most_similar_1 = similar_retrival(instance, instances_0,instances_1)
    print("this step 3")
    # Convert to list of dictionaries for JSON response
    def convert_to_feature_dict(instance):
        # return {f'feature_{i}': value for i, value in enumerate(instance)}
        return instance.to_dict()
    
    response = {
        'instance_0': convert_to_feature_dict(most_similar_0),
        'instance_1': convert_to_feature_dict(most_similar_1)
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
  
  
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',port=3333)
