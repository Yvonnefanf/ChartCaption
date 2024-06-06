from flask import Flask, request, jsonify,send_from_directory
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS, cross_origin
import json
from utils import org_instance_importance,get_model
import numpy as np
import random
import lime
import re
from datetime import datetime
from matplotlib.lines import Line2D
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

    img_path,features_with_values,features_values, features_, org_importance,features_with_bound = org_instance_importance(model, X_train, y_train,X, index)
   
    print("features_values",features_values)
    return jsonify({'img_path': img_path,'features_with_bound':features_with_bound,"value_list":features_values})
  
@app.route('/update_annotation', methods=['POST','GET'])
@cross_origin()
def update_annotation():
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
    new_importance = [abs(val) for val in exp_dict.values()]
  
    max_importance = max(new_importance)
    instance_str = instance.apply(lambda x: str(x))

    # Extract feature names and values using regular expressions
    modified_features_with_values = []
    modified_features = []
    for feature in features:
        match = re.findall(r'([a-zA-Z]+)', feature)
        # print("match",feature, match)
        if match:
            feature_name = ' '.join(match)  # Combine all matches with a space
            feature_value = instance_str[feature_name]
            modified_features_with_values.append(f"{feature_name}={feature_value}")
            modified_features.append(f"{feature_name}")
        else:
            modified_features_with_values.append(f"{feature}={instance_str[feature]}")
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)

        # Create a mapping from modified feature names to their new importance
    new_importance_dict = dict(zip(modified_features, new_importance))

    # Plot Feature Importance with values
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_yticklabels([])
    ax.yaxis.set_ticks([])
    bars = ax.barh(features_with_values, org_importance, color='skyblue',alpha=0.4, label='Original Importance')

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
    now=datetime.now()
    image_path = 'imgs/lime_feature_importance_newinstance_{}{}.png'.format(instance_index,now)
    plt.savefig(image_path,transparent=True)
    plt.close()
    return jsonify({'img_path': image_path,'features_with_bound':[],"value_list":[],"predict":int(predict[0])})
  
  
if __name__ == '__main__':
    app.run(debug=True, port=3333)
