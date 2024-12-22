# time-series-synthetic


## Contents
- [Install](#install)
- [Dataset](#dataset)
- [Models](#pre-trained-models)
- [Training](#training)
- [Generation](#generation)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to time-series-synthetic folder
```bash
git clone https://github.com/betterdataai/time-series-synthetic.git
cd time-series-synthetic
```

2. Install Package
```Shell
conda create -n ts
conda activate ts
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

## Pre-trained Models

| Model | #Params | Download |
|:---:|:---:|:---:|
| Walmart-TS| M | |






## Training
The training consists two stages. Stage1 is to train a VQVAE for discrete tokenization and Stage2 is to train a transformer model for TS generation.

### Stage1
Args you need to pass: train_data_path, test_data_path, static_cond_dim, seq_len

```
python stage1.py --use_custom_dataset True --dataset_names Walmart --train_data_path datasets/CustomDataset/Walmart_train.csv --test_data_path datasets/CustomDataset/Walmart_test.csv --static_cond_dim 6 --seq_len 143 --gpu_device_ind 0
```


### Stage2

```
python stage2.py --use_custom_dataset True --dataset_names Walmart --train_data_path datasets/CustomDataset/Walmart_train.csv --test_data_path datasets/CustomDataset/Walmart_test.csv --static_cond_dim 6 --seq_len 143 --gpu_device_ind 0
```




## Generation
```
python generation.py --use_custom_dataset True --dataset_names Walmart --train_data_path datasets/CustomDataset/Walmart_train.csv --test_data_path datasets/CustomDataset/Walmart_test.csv --static_cond_dim 6 --seq_len 143 --gpu_device_idx 0
```

## Evaluation
