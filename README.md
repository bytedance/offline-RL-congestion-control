## NOTES
This repo is for the Ten-TMS team to participate in the RL challenge. 

emulate_data_dir is supposed to located at "./emulated_dataset"
testbed_data_dir is supposed to located at "./testbed_dataset"
We also suppose a directory called data_dir for further data split which is "./" by default

Before training, the program will split the dataset into train, validation, test data, which will be moved to folders './emulated_dataset', './train_eval_dataset', './eval_dataset', respectively. 

We pre-store the behavior of baseline model for after-train evaluation. The file is baseline_eval.pkl, which is located at './emulated_dataset'. 

## 1. Dependencies
pip install -r requirements.txt

## 2. Generate Test Set and Validation Set
Setting the correct path and run test_set_generator.py 

## 3. Generate Pretrain Model
Set the pretrain_config.json, then run pretrain/train.py

## 4. Finetune Model
Set the finetune_config.json, the run fintune/train.py


