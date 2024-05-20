import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist

# 设置你的令牌和模型
token = "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"
############## llama 7b
model_name = "meta-llama/Llama-2-7b-chat-hf"
download_directory = "/data1/yifan/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"

############## llama 13b
# model_name = "meta-llama/Llama-2-13b-chat-hf"
# download_directory = "/data1/yifan/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"

# check CUDA
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. Please check your installation.")

device = torch.device("cuda:2")  # 使用 CUDA
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# model = AutoModelForCausalLM.from_pretrained(download_directory)
with torch.no_grad():  # Ensure no gradients are being computed
    model = AutoModelForCausalLM.from_pretrained(download_directory).half()  # Convert to FP16
    model.eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(download_directory)
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token,force_download=True,cache_dir='/data1/yifan')


messages = [
    {"role": "system", "content": "Identify the most influential feature based on absolute value and explain why."},
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

model.eval()

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

generated_ids = model.generate(input_ids, max_new_tokens=1000, do_sample=True)
outputs = tokenizer.batch_decode(generated_ids)

output_text = outputs[0]
response_start = output_text.rfind("[INST]")  # Find the last instruction block
response_end = output_text.rfind("</s>") if response_start != -1 else -1

# Extracting the specific response if indices are found
if response_start != -1 and response_end != -1 and response_end > response_start:
    response_text = output_text[response_start:response_end].strip()
else:
    response_text = "No valid response found."

print("Generated Response:", response_text)



