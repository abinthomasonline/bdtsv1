#DATA_PATH = "datasets/source_train.csv" # source dataset before preprocessing, csv file
DP_TRAIN_DATA_PATH="./datasets/dp_train.csv" # training dataset before data pipeline, must keep the exact path
DP_VAL_DATA_PATH="./datasets/dp_val.csv" #  validation dataset before data pipeline, must keep the exact path
DP_STATIC_COND_PATH="./datasets/dp_condition.csv" # condition dataset before data pipeline, must keep the exact path
TRAIN_DATA_PATH = "./datasets/train.csv" # training dataset for training, processed after data pipeline, must keep the exact path
VAL_DATA_PATH = "./datasets/val.csv" # validation dataset for training, processed after data pipeline, must keep the exact path
STATIC_COND_PATH = "./datasets/condition.csv" # synthetic conditions for TS data generation, processed after data pipeline, must keep the exact path
DATA_SAVE_PATH = "./datasets/"
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
	@echo "MODE is '$(MODE)'"
ifeq ($(MODE),validate1b)
	@echo "Learning Data Config"
	@mkdir -p $(OUTPUT_PATH)
	@python main.py prepare -d $(DATA_PATH) -p $(DATA_POLICY_PATH) -o $(DATA_CONFIG_PATH)
else
	@echo "Validating edited 2b JSON"
	@python3 main.py -t -l $(LOG_LEVEL) validate -d $(DATA_PATH) -p $(DATA_CONFIG_PATH_2B)
#	@cp $(DATA_CONFIG_PATH_2B) $(OUTPUT_PATH)/ts_data_config_learn.json
endif


train_ts:
	@mkdir -p $(MODEL_OUTPUT_PATH)
	@python main.py -t -l $(LOG_LEVEL) arf -d $(DATA_PATH) -p $(DATA_CONFIG_PATH) -m $(MODEL_CONFIG_PATH) -o $(DATA_SAVE_PATH)
	@python main.py -t -l $(LOG_LEVEL) preprocess -v False -c False -d $(DP_TRAIN_DATA_PATH) -p $(DATA_CONFIG_PATH) -m $(MODEL_CONFIG_PATH) -o $(DATA_SAVE_PATH)
	@python main.py -t -l $(LOG_LEVEL) preprocess -v True -c False -d $(DP_VAL_DATA_PATH) -p$(DATA_CONFIG_PATH) -o $(DATA_SAVE_PATH)
	@python main.py -t -l $(LOG_LEVEL) preprocess -v False -c True -d $(DP_STATIC_COND_PATH) -p$(DATA_CONFIG_PATH) -o $(DATA_SAVE_PATH)
	@python3 main.py -t -l $(LOG_LEVEL) train -d $(TRAIN_DATA_PATH) -v $(VAL_DATA_PATH) -m $(MODEL_CONFIG_PATH)

wrapup_train_ts:
	@cp ./saved_models/* $(MODEL_OUTPUT_PATH)
	@echo "Training Completed"

sample_ts:
	@mkdir -p $(GEN_DATA_OUTPUT_PATH)
	@python3 main.py -t -l $(LOG_LEVEL) sample -sd $(STATIC_COND_PATH) -m $(MODEL_CONFIG_PATH)

wrapup_sample_ts:
	@mv ./synthetic_data/* $(GEN_DATA_OUTPUT_PATH)
	@echo "Sampling Completed"

