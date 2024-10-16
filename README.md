# MergeMultiModalityLM

## Load Merged Models 
[DeSTA2](https://huggingface.co/DeSTA-ntu/DeSTA2-8B-beta)
```
merged_state_dict = torch.load(merge_model_path)
desta = AutoModel.from_pretrained("DeSTA-ntu/DeSTA2-8B-beta", trust_remote_code=True, revision="502e7dd95f802a4d9885cf1b4cf37462ad250ac8")
desta_lm = desta.llama
desta_lm.model.layers.load_state_dict(merged_state_dict)
```
[Llava3](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf)
```
merged_state_dict = torch.load(merge_model_path)
llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
llava = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch.float16)
llava_lm = llava.language_model
llava_lm.model.layers.load_state_dict(merged_state_dict)
```