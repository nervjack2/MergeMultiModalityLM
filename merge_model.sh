#!/bin/bash

# 檢查參數是否正確
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <merge_config_dir> <save_dir> <python_script>"
    exit 1
fi

# 指定參數
merge_config_dir=$1
save_dir=$2
python_script=$3  # The Python script to execute

# 如果 save_dir 不存在，則建立此目錄
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
    echo "Directory $save_dir created."
fi

# 遍歷 merge_config_dir 中的所有 .yaml 檔案
for config_file in "$merge_config_dir"/*.yaml; do
    # 獲取檔案名稱，不包含副檔名
    config_name=$(basename "$config_file" .yaml)
    
    # 設定 save_path 和 merge_config_path
    save_path="$save_dir/${config_name}.ckpt"
    merge_config_path="$config_file"
    
    # 執行 Python 程式，使用指定的腳本
    ~/anaconda3/envs/MergeLLM/bin/python3 "$python_script" "$save_path" "$merge_config_path"
    
    # 檢查指令是否成功
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing $config_file"
        exit 1
    fi
done

echo "All files processed successfully."
