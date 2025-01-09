#DATA_PATH = "datasets/CustomDataset/source_train.csv" # source dataset before preprocessing, csv file
#VAL_DATA_PATH="datasets/CustomDataset/source_val.csv" # source validation dataset before preprocessing, csv file
DATA_SAVE_PATH = "datasets/CustomDataset"
#STATIC_COND_PATH="datasets/CustomDataset/static_cond.csv" # static conditions for TS data generation
DATA_POLICY_PATH="./configs/data/data_config.json"# 1a config file
OUTPUT_PATH="./out"# output directory
DATA_CONFIG_PATH="${OUTPUT_PATH}/data-config.json"  # learned and edited 1b config file
#DATA_CONFIG_PATH_2A=$(OUTPUT_PATH)/ts_data_config_learn.json# learned and edited 1b config file
MODEL_CONFIG_PATH="./configs/model/config.json"# model parameters
# LOG_LEVEL="INFO"
# MODE=validate1b# OR "validate2b"  #validate1b for 4A, validate2b for 4B step.
MODEL_OUTPUT_PATH="$(OUTPUT_PATH)/model"
GEN_DATA_OUTPUT_PATH="$(OUTPUT_PATH)/data"
# DATA_CONFIG_PATH_2B="/tmp/data_args_1a.json"  #learned 4A data config json for 4B.


prepare_ts:
	@mkdir -p $(OUTPUT_PATH)
	@python main.py prepare -d $(DATA_PATH) -p $(DATA_POLICY_PATH) -o $(DATA_CONFIG_PATH)

train_ts:
	@mkdir -p $(MODEL_OUTPUT_PATH)
	@python main.py -t -l $(LOG_LEVEL) preprocess -v False -d $(DATA_PATH) -p $(DATA_POLICY_PATH) -m $(MODEL_CONFIG_PATH)-o $(DATA_SAVE_PATH)
	@python main.py -t -l $(LOG_LEVEL) preprocess -v True -d $(VAL_DATA_PATH) -p$(DATA_POLICY_PATH) -o $(DATA_SAVE_PATH)
	@python3 main.py -t -l $(LOG_LEVEL) train \
      -d $(DATA_PATH) -v $(VAL_DATA_PATH) -m $(MODEL_CONFIG_PATH)

wrapup_train_ts:
	@mv ./saved_models/* $(MODEL_OUTPUT_PATH)
	@echo "Training Completed"

sample_ts:
	@mkdir -p $(GEN_DATA_OUTPUT_PATH)
	@python3 main.py -t -l $(LOG_LEVEL) sample \
      -d $(DATA_PATH) -sd $(STATIC_COND_PATH) -m $(MODEL_CONFIG_PATH)

wrapup_sample_ts:
	@mv ./synthetic_data/* $(GEN_DATA_OUTPUT_PATH)
	@echo "Sampling Completed"

