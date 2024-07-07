from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def get_rules(pipeline,X_train,X, shap_binary_labels,preprocessor,instance_index=0):
    # Select a specific instance (e.g., the first instance in the test set)

    instance_scaled = pipeline.named_steps['preprocessor'].transform(X_train.iloc[[instance_index]])
    instance_original = X_train.iloc[instance_index].values.reshape(1, -1)

    feature_names = [
        'cap-diameter',  
        'does-bruise-or-bleed',
        'stem-height',
        'stem-width',
        'has-ring',
        'season'
    ]    
    # Identify categorical features
    categorical_features = ['does-bruise-or-bleed', 'has-ring', 'season']
    numerical_features = [col for col in X.columns if col not in categorical_features]


    # Initialize the rules dictionary
    rules_dict = {feature_name: [] for feature_name in feature_names}

    # Get the decision path for the instance and generate natural language explanations
    multi_target_clf = pipeline.named_steps['classifier']

    # Identify numerical feature indices
    numerical_features_indices = [X.columns.get_loc(feature) for feature in numerical_features]
    cat_features_indices = [X.columns.get_loc(feature) for feature in categorical_features]

    for i, estimator in enumerate(multi_target_clf.estimators_):
        node_indicator = estimator.decision_path(instance_scaled)
        leaf_id = estimator.apply(instance_scaled)

        feature_indices = estimator.tree_.feature

        # All nodes in the decision path
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

        print(f"Explanation for feature {feature_names[i]} (binary SHAP value prediction):")
        for node_id in node_index:
            if leaf_id == node_id:
                continue

            # Get the relevant information for the decision node
            feature_index = feature_indices[node_id]
            if feature_index == -2:
                continue

            if feature_index in numerical_features_indices:  # Numerical feature
                feature_name = X.columns[feature_index]
                feature_value_scaled = instance_scaled[0, feature_index]
                feature_value_original = instance_original[0, feature_index]

                # Inverse transform to get the original threshold value
                threshold_value_scaled = estimator.tree_.threshold[node_id]
                threshold_value_array = np.zeros((1, len(numerical_features)))
                threshold_value_array[0, numerical_features_indices.index(feature_index)] = threshold_value_scaled
                threshold_value_original = preprocessor.named_transformers_['num'].inverse_transform(threshold_value_array)[0, numerical_features_indices.index(feature_index)]

                if feature_value_original <= threshold_value_original:
                    threshold_sign = "<="
                    contribution = "negative" if shap_binary_labels[instance_index, i] == 0 else "positive"
                else:
                    threshold_sign = ">"
                    contribution = "positive" if shap_binary_labels[instance_index, i] == 1 else "negative"
            elif feature_index in cat_features_indices:  # One-hot encoded categorical feature
                category_index = np.where(cat_features_indices == feature_index)[0][0]

                one_hot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

                one_hot_feature_name = one_hot_feature_names[category_index]
                feature_name, category_name = one_hot_feature_name.split('_', 1)
                feature_value_original = instance_original[0, list(X.columns).index(feature_name)]
                threshold_sign = "="
                threshold_value_original = category_name
                # print("one_hot_feature_names111",original_feature_name,threshold_value_original)
                contribution = "positive" if instance_scaled[0, feature_index] == 1 else "negative"

            # Add the rule to the dictionary
            rules_dict[feature_names[i]].append((feature_name, feature_value_original, threshold_sign, threshold_value_original, contribution))

            # Generate the natural language explanation
            explanation = (
                f"For feature '{feature_name}', the value {feature_value_original} {threshold_sign} {threshold_value_original}. "
                f"This decision contributes to a {contribution} SHAP value for feature {i}."
            )
            print(explanation)

        print("\n")
    return rules_dict