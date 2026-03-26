## Train ED lora with private date

```bash
cd tools/Mix-of-Show
```
Step 1: generate the description for each character
```bash
bash Script/Step_1_generate_desc_char.sh
```

Step 2: Generate the data .json file and .json file  for each Character to train the lora
```bash
bash Script/Step_2_generate_data_json_for_character.sh
```

Step 3: train lora for each character
```bash
export PYTHONPATH=/storage/wuweijia/MovieGen/MovieDirector/MovieDirector/tools/Mix-of-Show:$PYTHONPATH 

for file in /storage/wuweijia/MovieGen/MovieDirector/MovieDirector/tools/Mix-of-Show/options/train/EDLoRA/MovieGen/InsideOut2/*.yml; do
    CUDA_VISIBLE_DEVICES=2 accelerate launch train_edlora.py -opt "$file"
done
```


Step 4: Generate config .json file for multi lora
```bash
bash Script/Step_4_generate_config_json_for_multi_lora.sh
```

Step 5: train multi lora 

```bash
bash train_lora_multi.sh
```

Step 6: Test

```bash
CUDA_VISIBLE_DEVICES=4  python sample_single_for_fusion.py
```