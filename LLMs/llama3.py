import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist

# 设置你的令牌和模型
token = "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"
############## llama 7b
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
download_directory = "/data1/yifan/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=token,cache_dir='/data1/yifan')
# Check if the tokenizer has an EOS token, if not, set it up
# tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # If no EOS token, add a new pad token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
         
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_auth_token=token,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir='/data1/yifan'
)
# Prepare input

messages = [
    {"role": "system", "content": "Answer the question"},
    {"role": "user", "content": 
        """The feature values are: 
        Feature: Age, Value: -1.47
        Feature: Occupation, Value: -0.57
        Feature: Relationship, Value: 1.31
        Feature: Capital Gain, Value: -0.17
        Feature: Education-Num, Value: 0.57
        What is the feature determined the instance's prediction"""
    },
    {"role": "assistant", "content": 
        """ Most influential feature: Age, the value of this feature is -1.47, which is the highest absolute value in all the features."""
    },
    {"role": "user", "content": 
        """The feature values are: 
        Feature: Age, Value: -1.59
        Feature: Workclass, Value: 0.22
        Feature: Marital Status, Value: 0.09
        Feature: Occupation, Value: -0.57
        Feature: Relationship, Value: 1.91
        Feature: Capital Gain, Value: -0.17
        Feature: Education-Num, Value: 0.57
        What is the feature determined the instance's prediction"""
    },
    {"role": "assistant", "content": 
        """ Most influential feature: Relationship,  the value of this feature is 1.91, which is the highest absolute value in all the features."""
    },
    {"role": "user", "content": 
        """The feature values are: 
                Feature: Years of Education, Value: 16
                Feature: Marital Status, Value: 45
                Feature: Capital Gain, Value: -56
                Feature: Capital Loss, Value: -22
                Feature: Hours Per week, Value: -17
                Feature: Others, Value: 45,
        What is the feature determined the instance's prediction"""
    },
]

# Combine previous examples and current question into a single string
context_plus_query = "\n".join([msg['content'] for msg in messages[:-1]])  # Exclude the last query

# Tokenize texts
input_dict = tokenizer(context_plus_query, return_tensors="pt", padding=True)
input_ids = input_dict['input_ids'].to(model.device)
attention_mask = input_dict['attention_mask'].to(model.device)

# Prepare the context and question from previous and current messages
messages = [
    {"role": "system", "content": "You need generate caption for the chart data"},
    # {"role": "user", "content": 
    #     """Here are the SHAP values for each feature:  
    #     Age: -1.47
    #     Occupation: -0.57
    #     Relationship: 1.31
    #     Capital Gain: -0.17
    #     Education-Num: 0.57
    #     What is the feature determined the instance's prediction"""
    # },
    # {"role": "assistant", "content": 
    #     "Most influential feature: Age, the SHAP value of this feature is -1.47, which is the highest absolute value in all the features."
    # },
    # {"role": "user", "content": 
    #     """Here are the SHAP values for each feature: 
    #     Age: -1.59
    #     Workclass: 0.22
    #     Marital Status: 0.09
    #     Occupation: -0.57
    #     Relationship: 1.91
    #     Capital Gain: -0.17
    #     Education-Num: 0.57
    #     What is the most influential feature based on these SHAP values?"""
    # },
    # {"role": "assistant", "content": 
    #     "Most influential feature: Relationship, the SHAP value of this feature is 1.91, which is the highest absolute value in all the features."
    # },
    {"role": "user", "content": 
        """Here are the SHAP values for each feature:
        Years of Education:16
        Capital Gain: -56
        Marital Status: -45
        Capital Loss: -22
        Hours Per week: -17
        Others: 45,
        please generate caption for this data?"""
    },
]

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    eos_token_id=terminators,
    do_sample=True,
    top_p=0.9,
    temperature=0.6
)
response = outputs[0][input_ids.shape[-1]:]
print('output::',tokenizer.decode(response, skip_special_tokens=True))