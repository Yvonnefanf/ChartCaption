from flask import Flask, request, jsonify,send_from_directory
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS, cross_origin
import json
import datetime
from utils import load_model_and_tokenizer, initialize_device


MODEL_DIRECTORY = "/mnt/raid/yifan/LLMs/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590" # 7B
# MODEL_DIRECTORY = "/mnt/raid/yifan/LLMs/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8" # 13B
DEVICE = initialize_device(cuda_index=1)

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(MODEL_DIRECTORY, DEVICE)


app = Flask(__name__,static_folder='Frontend')
cors = CORS(app, supports_credentials=True)

def caption_generation(info):
    messages = [
            {"role": "system", "content": """Generate a caption for a bar chart displaying SHAP values. 
             Please provide a clear and concise summary of the key insights from this visualization in 6 sentences."""
            },
            # {"role": "user", "content": 
            #     """Here are the SHAP values for each feature:  
            #     Age: -1.47
            #     Occupation: -0.57
            #     Relationship: 1.31
            #     Capital Gain: -0.17
            #     Education-Num: 0.57
            #    Please generate caption for this data"""},
            # {  
            #    "role": "assistant", "content": 
            #     """
            #     Model analysis reveals the varying impact of individual features on the prediction outcome. 
            #     'Relationship' has the most positive influence with a SHAP value of 1.31. In contrast, 'Age' notably decreases the prediction score with a SHAP value 
            #     of -1.47.
            #     """
            # },
            # {"role": "user", "content": 
            #     """Here are the SHAP values for each feature:  
            #     Age: -1.59
            #     Workclass: 0.22
            #     Occupation: -0.57
            #     Relationship: 1.91
            #     Education-Num: 0.57
            #    Please generate caption for this data"""},
            # {  
            #    "role": "assistant", "content": 
            #     """
            #     Model analysis reveals the varying impact of individual features on the prediction outcome. 
            #     'Education-Num' has the most positive influence with a SHAP value of 0.57. In contrast, 'Relationship' notably decreases the prediction score with a SHAP value 
            #     of -1.91.
            #     """
            # },
            {"role": "user", "content": 
                """Please generate a caption for a dataset predicting 'High Income' or 'Low Income.' 
                The instance currently predicts 'High Income.'
                Feature Values are:
                Years of Education: 13 years
                Marital Status: married
                Capital Gain: 0
                Capital Loss: 0
                Hours Per week: 40 hours
                and SHAP values are: 
                Years of Education: 16
                Marital Status: 45
                Capital Gain: -56
                Capital Loss: -22
                Hours Per week: -17
                Others: 45, 
                Positive values indicate a prediction of 'High Income,' while negative values suggest 'Low Income.'"
                please highlight {info} into new caption.
                Please generate caption for this data.
                """
            },
        ]
    MAX_NEW_TOEKN = 500
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOEKN, do_sample=True)
    outputs = tokenizer.batch_decode(generated_ids)
    print("finfished ask.....")
    output_text = outputs[0]
    response_start = output_text.rfind("[/INST]")  # Find the last instruction block
    response_end = output_text.rfind("</s>") if response_start != -1 else -1

    # Extracting the specific response if indices are found
    if response_start != -1 and response_end != -1 and response_end > response_start:
        response_text = output_text[response_start:response_end].strip()
    else:
        response_text = "No valid response found."
    return response_text

def answer_question(question):
    print("start ask.....")
    if question == "What is the most influential feature based on these SHAP values?":
        print('type one')
        
        # Prepare the context and question from previous and current messages
        messages = [
            {"role": "system", "content": "Analyze the feature importances based on SHAP values and determine the most influential feature."},
            {"role": "user", "content": 
                """Here are the SHAP values for each feature:  
                Age: -1.47
                Occupation: -0.57
                Relationship: 1.31
                Capital Gain: -0.17
                Education-Num: 0.57
                What is the feature determined the instance's prediction"""
            },
            {"role": "assistant", "content": 
                "Most influential feature: Age, the SHAP value of this feature is -1.47, which is the highest absolute value in all the features."
            },
            {"role": "user", "content": 
                """Here are the SHAP values for each feature: 
                Age: -1.59
                Workclass: 0.22
                Marital Status: 0.09
                Occupation: -0.57
                Relationship: 1.91
                Capital Gain: -0.17
                Education-Num: 0.57
                What is the most influential feature based on these SHAP values?"""
            },
            {"role": "assistant", "content": 
                "Most influential feature: Relationship, the SHAP value of this feature is 1.91, which is the highest absolute value in all the features."
            },
            {"role": "user", "content": 
                """Here are the SHAP values for each feature:
                Years of Education:16
                Capital Gain: -56
                Marital Status: -45
                Capital Loss: -22
                Hours Per week: -17
                Others: 45,
                {question}?"""
            },
        ]
        MAX_NEW_TOEKN = 200
    else:
        print('other')
        messages = [
            {"role": "system", "content": "Please answer the question."},
            {"role": "user", "content": 
                """Here are the SHAP values for each feature:
                Years of Education:16
                Capital Gain: -56
                Marital Status: -45
                Capital Loss: -22
                Hours Per week: -17
                Others: 45,
                current prediction result is positive,
                {question}"""
            },
        ]
        MAX_NEW_TOEKN = 500
    
 
    
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOEKN, do_sample=True)
    outputs = tokenizer.batch_decode(generated_ids)
    print("finfished ask.....")
    output_text = outputs[0]
    response_start = output_text.rfind("[/INST]")  # Find the last instruction block
    response_end = output_text.rfind("</s>") if response_start != -1 else -1

    # Extracting the specific response if indices are found
    if response_start != -1 and response_end != -1 and response_end > response_start:
        response_text = output_text[response_start:response_end].strip()
    else:
        response_text = "No valid response found."
    return response_text

@app.route('/get_answer', methods=['POST'])
@cross_origin()
def get_answer():
    data = request.get_json()
    print("data",data)
    question = data.get('question') 
    print("question",question)
    if question:
        response = answer_question(question)
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        msg = {'text': response, 'time': formatted_time, 'sender': 'robot'}
        messages = load_messages()  # Load existing messages
        messages.append(msg)  # Append the new message
        
        with open(msg_history_path, 'w') as file:
            json.dump(messages, file, indent=4)  # Save the updated list of messages
        return jsonify({'msglist': messages})
    else:
        return jsonify({'error'})

msg_history_path = '../Data/ChatHistory/admin.json'
@app.route('/save_message', methods=['POST'])  # Changed to GET for simplicity
def save_message():
    data = request.get_json()
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    message = {'text': data.get('question'), 'time': formatted_time, 'sender': 'user'}
    messages = load_messages()  # Load existing messages
    messages.append(message)  # Append the new message
    with open(msg_history_path, 'w') as file:
        json.dump(messages, file, indent=4)  # Save the updated list of messages
    return jsonify({'msglist': messages}) 
    # return jsonify({"status": "Message saved"})

def load_messages():
    try:
        with open(msg_history_path, 'r') as file:
             return json.load(file)  # Load and return the list of messages
    except FileNotFoundError:
        print("File not found. Returning an empty list.")
        return []  # Return an empty list if the file does not exist


@app.route('/get_historymsg', methods=['GET'])  # Changed to GET for simplicity
@cross_origin()
def get_historymsg():
    messages = load_messages()
    return jsonify({'msglist': messages}) 

@app.route('/caption_gen', methods=['POST'])
@cross_origin()
def caption_gen():
    data = request.get_json()
    info = data.get('info') 
    
    caption = caption_generation(info)
    return jsonify({'caption': caption}) 

@app.route("/", methods=["GET", "POST"])
def GUI():
    # return render_template("SilasIndex.html")
    return send_from_directory(app.static_folder, 'index.html')

  

if __name__ == '__main__':
    app.run(debug=True, port=6001)
