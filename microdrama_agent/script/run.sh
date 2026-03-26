export CUDA_VISIBLE_DEVICES=0
# open ai key 
export OPENAI_API_KEY="sk-351a57bf5de445febf8c445879ade9a0"

# open ai key (aliyun, deepseek r1)
# export OPENAI_API_KEY=

# LLM: gpt4-o | deepseek-r1 | deepseek-v3 | qwen2.5-72b-instruct | llama3.3-70b-instruct
# gen_model: ROICtrl | StoryDiffusion
# Image2Video: SVD | I2Vgen | CogVideoX | Wan2.1 | HunyuanVideo_I2V
CUDA_VISIBLE_DEVICES=0 python3 run.py \
    --script_path ../dataset/FrozenII/script_synopsis.json \
    --character_photo_path ../dataset/FrozenII/character_list \
    --LLM deepseek-v3 \
    --gen_model ROICtrl \
    --audio_model VALL-E \
    --talk_model Hallo2\
    --Image2Video HunyuanVideo_I2V


