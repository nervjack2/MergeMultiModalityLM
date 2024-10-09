from transformers import  AutoModel

model = AutoModel.from_pretrained("DeSTA-ntu/DeSTA2-8B-beta", trust_remote_code=True, revision="502e7dd95f802a4d9885cf1b4cf37462ad250ac8")
model.to("cuda")
outputs = model.chat(
    [
        {"role": "system", "content": "Focus on the speech and answer the questions."},
        {"role": "audio", "content": "/media/nervjack2/376c3409-19c5-44ba-942d-5155cc0cdfc5/dataset/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"},
        {"role": "user", "content": "說說這段聲音中有什麼特點，請告訴我他的性別"},
    ], max_new_tokens=256, do_sample=True, temperature=1, top_p=1
)

print(model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])