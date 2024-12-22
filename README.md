# time-series-synthetic


## Contents
- [Install](#install)
- [Dataset](#dataset-preperation)
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

## Dataset Preperation
As of 12/22/2024, the conditional TS model only supports .csv file as the input data. All dataset should be in ".csv" form and follow the format below:
```
Num_Col = Num_Static_Cond (after trasformation using Betterdata data pipeline) + Num_TS_Channel
```
### Walmart Dataset Example:
#### Orignial static conditions:
<img width="581" alt="image" src="https://github.com/user-attachments/assets/1200f382-1db1-4c1c-b48e-5b25e719af49" />

#### Static conditions after transformation:
<img width="927" alt="image" src="https://github.com/user-attachments/assets/2e36f2c2-b4dc-4987-b5f2-ad60725a4cf5" />

#### Traning data should look like:
<img width="1075" alt="image" src="https://github.com/user-attachments/assets/81d572c3-bec6-47c8-8ae9-080e284792d9" />






## Pre-trained Models

| Model | #Params | Num_Static_Cond| Static_Cond_Dim | Num_TS_Channel (# of TS Features) | Seq_Length | Download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Walmart-TS| 1.3M | 4 | 6 | 1 | 143 |[model](https://github.com/betterdataai/time-series-synthetic/blob/main/saved_models/stage2-Walmart.ckpt)|






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
