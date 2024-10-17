import torch
import sys
import copy
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from merge_utils import MergeTools
from qwen_vl_utils import process_vision_info

save_path, merge_config_path = sys.argv[1], sys.argv[2]
with open(merge_config_path, "r", encoding="utf-8") as file:
    merge_config = yaml.safe_load(file)
# Load Qwen 7B
qwenlm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True, torch_dtype=torch.float16)
# Load Qwen2 Audio
qwena_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
qwena_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True, torch_dtype=torch.float16)
# Load Qwen Vision
qwenv_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
qwenv_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True, torch_dtype=torch.float16)
# Discard embed tokens and output projection matrix for merging, since they have different number of parameters 
qwena_state_dict = copy.deepcopy(qwena_model.transformer.h.state_dict())
qwenv_state_dict = copy.deepcopy(qwenv_model.transformer.h.state_dict())
qwenlm_state_dict = copy.deepcopy(qwenlm.transformer.h.state_dict())
# Merge LM
print(f"Merging Qwen Audio and Qwen Vison with merging configuration = {merge_config}")
merge_tools = MergeTools(merge_config)
merged_state_dict = merge_tools.merge(qwenv_state_dict, qwena_state_dict, qwenlm_state_dict)
# Save Merged State Dictionary
torch.save(merged_state_dict, save_path)
# Testing Merged Parameters for Qwen Audio
print("Testing Merged Model")
qwena_model.transformer.h.load_state_dict(merged_state_dict)
qwena_model.to("cuda").eval()
query = qwena_tokenizer.from_list_format([
    {'audio': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.flac'}, # Either a local path or an url
    {'text': 'what does the person say?'},
])
response, history = qwena_model.chat(qwena_tokenizer, query=query, history=None)
print(response)
# The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".
# 2nd dialogue turn
response, history = qwena_model.chat(qwena_tokenizer, 'Find the start time and end time of the word "middle classes"', history=history)
print(response)
# The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.

# Testing Merged Parameters for Qwen Vision
qwenv_model.transformer.h.load_state_dict(merged_state_dict)
qwenv_model.to("cuda").eval()
# 1st dialogue turn
query = qwenv_tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'text': '这是什么'},
])
response, history = qwenv_model.chat(qwenv_tokenizer, query=query, history=None)
print(response)
# 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种可能是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，似乎在和人类击掌。两人之间充满了信任和爱。