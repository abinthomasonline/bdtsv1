VAL_DATA_PATH="datasets/CustomDataset/val.csv" # validation dataset
STATIC_COND_PATH="datasets/CustomDataset/static_cond.csv" # static conditions for TS data generation
DATA_POLICY_PATH="./configs/data/gan_core.json"# 1a config file
OUTPUT_PATH="./out"# output directory
DATA_CONFIG_PATH_2A=$(OUTPUT_PATH)/ts_data_config_learn.json# learned and edited 1b config file
MODEL_CONFIG_PATH="./configs/model/config.json"# model parameters
# LOG_LEVEL="INFO"
# MODE=validate1b# OR "validate2b"  #validate1b for 4A, validate2b for 4B step.
MODEL_OUTPUT_PATH=$(OUTPUT_PATH)/model
GEN_DATA_OUTPUT_PATH=$(OUTPUT_PATH)/data
# DATA_CONFIG_PATH_2B="/tmp/data_args_1a.json"  #learned 4A data config json for 4B.

data_validation:
	@echo "MODE is '$(MODE)'"
ifeq ($(MODE),validate1b)
	@echo "Learning Data Config"
	@mkdir -p $(OUTPUT_PATH)
	@python3 main.py -t -l $(LOG_LEVEL) prepare \
		-d $(DATA_PATH) -p $(DATA_POLICY_PATH) -o $(DATA_CONFIG_PATH_2A)
else
	@echo "Validating edited 2b JSON"
	@mkdir -p $(OUTPUT_PATH)
# don't understand this part
	@python3 main.py -t -l $(LOG_LEVEL) validate \
		-d $(DATA_PATH) -p $(DATA_CONFIG_PATH_2A)
# 	@cp $(DATA_CONFIG_PATH_2B) $(OUTPUT_PATH)/ts_data_config_learn.json
endif


train_ts:
	@mkdir -p $(OUTPUT_PATH)
	@python3 main.py -t -l $(LOG_LEVEL) train \
      -d $(DATA_PATH) -v $(VAL_DATA_PATH) -m $(MODEL_CONFIG_PATH)

wrapup_model:
    @mkdir -p $(MODEL_OUTPUT_PATH)
	@mv ./saved_models/* $(MODEL_OUTPUT_PATH)
	@echo "Training Completed"

sample_ts:
	@python3 main.py -t -l $(LOG_LEVEL) sample \
      -d $(DATA_PATH) -sd $(STATIC_COND_PATH) -m $(MODEL_CONFIG_PATH)

wrapup_data:
    @mkdir -p $(GEN_DATA_OUTPUT_PATH)
    @mv ./synthetic_data/* $(GEN_DATA_OUTPUT_PATH)
    @echo "Sampling Completed"

