# Data Processing

## 1. Process Scientific Pre-training Dataset

### 1.1 Process Chinese Dataset

```bash
cd pretrain/chinese_process/
python collect.py
```

### 1.2 Process English Dataset

```bash
cd pretrain/english_process/
python collect.py
```

Note that the `redpajama_train.json` is obtained by running the following command:
```bash
cd pretrain
python download_from_hf.py
```

### 1.3 Combine All the Dataset

First of all, get into the folder `train`, and run combine the chinese and english datasets and get the `train.txt` file:
```bash
./combine_chinese_corpus.sh
```

Then, split the `train.txt` into 8 subfiles (each for one GPU process to load):
```bash
# shuffle and split
./split.sh
```

## 2. Process SFT Dataset

### 2.1 Process Paper-ground Question Answering Dataset

Download the  QASPER-v0.3 dataset from the path listed in `../README.md` and run:
```bash
python collect.py
```

### 2.2 Process Emotional Dialogue Dataset

Simple download the scientific emotional dialogue dataset and put it under the `./data/sft/emotional` folder.

### 2.3 Process Dolly Dataset

Download Dolly corpus by run:
```bash
cd data/sft/dolly
python download_from_hf.py
```

### 2.4 Process SciMRC Dataset

Download SciMRC dataset by the link listed in `../README`, and process it by running the following command:
```bash
python collect.py
```

### 2.5 Combine Paper-ground Instruction Dataset

Combine the dolly, SciMRC and QASPER instruction dataset for supervised fine-tuning (without emotional dialogue dataset):
```bash
python combine.py
```
