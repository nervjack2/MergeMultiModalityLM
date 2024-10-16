import torch
import requests
import requests 
import sys 
import os 
import copy
import yaml
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import  AutoModel, AutoModelForCausalLM
from PIL import Image
from merge_utils import MergeTools
from tqdm import tqdm

save_path, merge_config_path = sys.argv[1], sys.argv[2]
with open(merge_config_path, "r", encoding="utf-8") as file:
    merge_config = yaml.safe_load(file)
# Load Llava3 
llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
llava = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch.float16)
llava_lm = llava.language_model
# Load Desta
desta = AutoModel.from_pretrained("DeSTA-ntu/DeSTA2-8B-beta", trust_remote_code=True, revision="502e7dd95f802a4d9885cf1b4cf37462ad250ac8")
desta_lm = desta.llama
# Load Llama3 Instruct 8B
llama = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype=torch.float16)
# Discard embed tokens and output projection matrix for merging, since they have different number of parameters 
llava_lm_state_dict = copy.deepcopy(llava_lm.model.layers.state_dict())
desta_lm_state_dict = copy.deepcopy(desta_lm.model.layers.state_dict())
llama_state_dict = copy.deepcopy(llama.model.layers.state_dict())

print(f"Merging DeSTA2 and Llava3 with merging configuration = {merge_config}")
# Merge LM
merge_tools = MergeTools(merge_config)
merged_state_dict = merge_tools.merge(llava_lm_state_dict, desta_lm_state_dict, llama_state_dict)
# Save Merged State Dictionary
torch.save(merged_state_dict, save_path)
# Testing Merged Parameters for DeSTA2
print("Testing Merged Model")
desta_lm.model.layers.load_state_dict(merged_state_dict)
desta.to("cuda")
outputs = desta.chat(
    [
        {"role": "system", "content": "Focus on the speech and answer the questions."},
        {"role": "audio", "content": "/livingrooms/nervjack2/dataset/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"},
        {"role": "user", "content": "說說這段聲音中有什麼特點，請告訴我他的性別"},
    ], max_new_tokens=256, do_sample=True, temperature=1, top_p=1
)
print(desta.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
# Testing Merged Parameters for Llava3
llava_lm.model.layers.load_state_dict(merged_state_dict)
llava.to("cuda")
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
conversation = [
    {
    "role": "user",
    "content": [
        {"type": "text", "text": "What is shown in this image?"},
        {"type": "image"},
        ],
    },
]
prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = llava_processor(images=image, text=prompt, return_tensors="pt").to(llava.device)
output = llava.generate(**inputs, max_new_tokens=100)
print(llava_processor.decode(output[0], skip_special_tokens=True))