# Mozi: A Scientific Large-scale Language Model

![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

![墨子](./墨子_avatar.png)


**Team of Beijing Institute of Technology:** [Tian Lan](https://github.com/gmftbyGMFTBY)<sup>\*</sup>, Tianyi Che*, [Zewen Chi](https://github.com/CZWin32768), and Xian-ling Mao

****

<span id='all_catelogue'/>

## Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#environment'>2. Running Mozi Demo</a>
    * <a href='#install_environment'>2.1. Environment Installation</a>
    * <a href='#download_mozi_model'/>2.2. Prepare Mozi Checkpoint</a>
    * <a href='#running_demo'>2.5. Deploying Demo</a>
* <a href='#train_pandagpt'>3. Train Your Own Mozi models</a>
    * <a href='#data_preparation'>3.1. Data Preparation</a>
    * <a href='#training_configurations'>3.2. Training Configurations</a>
    * <a href='#model_training'>3.3. Training Mozi</a>
      * <a href='#scientific_pretraining'>3.3. Scientific Pre-Training Mozi Models</a>
      * <a href='#paper_ground_training'>3.3. Supervised Fine-tuning Mozi Models</a>
* <a href='#citation'>Citation</a>
* <a href='#technicalreport'>Technical Report</a>
* <a href='#acknowledgments'>Acknowledgments</a>

****

<span id='introduction'/>

### 1. Introduction: <a href='#all_catelogue'>[Back to Top]</a>


Mozi is the first large-scale language model for the scientific paper domain, such as question answering and emotional support.
With the help of the large-scale language and evidence retrieval models, SciDPR, Mozi generates concise and accurate responses to users' questions about specific papers and provides emotional support for academic researchers.

We will explore more real-world application scenarios for Mozi in the future, making it a foundation model for solving various scientific tasks. 

****

<span id='environment'/>

### 2. Running Mozi Models: <a href='#all_catelogue'>[Back to Top]</a>

<span id='install_environment'/>

#### 2.1. Environment Installation:
To install the required environment, please run
```
pip install -r requirements.txt
```

Then install the Pytorch package with the correct cuda version, for example
```
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch/
```

<span id='download_mozi_model'/>

#### 2.2. Prepare Mozi Checkpoint:
The Mozi model weights (pre-trained on scientific corpus) consists of the pre-trained large-scale language and the LoRA weights.

1. First of all, please download [LLaMA-7B checkpoint](https://huggingface.co/decapoda-research/llama-7b-hf) and [Baichuan-7B checkpoint](https://huggingface.co/baichuan-inc/baichuan-7B).
2. Then, please download the LoRA weights for these two models from:

    | LoRA checkpoints | Huggingface Delta Weights Address |
    | ---------------- | --------------------------------- |
    | Baichuan-7B delta weight | []() |
    | LLaMA-7B delta weight | []() |

3. We also release the delta LoRA model weights for scientific emotional dialogue, which can be found in [here](). The emotional dialogue delta weights are built on Baichuan-7B model. In the future, we will directly optimize this scientific emotional dialogue instruction tuning dataset with other instruction dataset, such as paper-ground question answering and scientific information retrieval.

Now, the model parameters are all prepared.

<span id='running_demo'/>

#### 2.3. Deploying Demo:
Upon completion of previous steps, you can run the demo locally as
```bash
./scripts/deploy.sh

# #!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python deploy.py \
#     --model scillm-sft\
#     --model_path baichuan-inc/baichuan-7B\
#     --delta_model_path ../ckpt/scillm-emotional-sft/18\
#     --port 23333
```

This script runs the Mozi model emotional model on `23333` port and the input `POST` request should be like:

```json
{
    "decoding_method": "greedy",
    "top_p": 0.7,
    "top_k": 10,
    "penalty_alpha": 0.5,
    "max_new_tokens": 128,
    "history": [
        "Human: 最近科研压力真的好大啊"
    ]
}
```

If you want to test the paper-ground dialog model, please replace the `--delta_model_path` with the corresponding model checkpoints weights that you download.
For the paper-ground dialog, the input `POST` request should be like:

```json
{
    "decoding_method": "greedy",
    "top_p": 0.7,
    "top_k": 10,
    "penalty_alpha": 0.5,
    "max_new_tokens": 128,
    "evidences": [
        "During the first two decades of the 21st century, the sharing and processing of vast amounts of data has become pervasive ...",
        "One way of circumventing this problem is to anonymise the data by removing, ...",
        "Given that this paper is concerned with text documents (e.g. medical records), the involved techniques are related to Natural Language Processing (NLP) ..."
    ],
    "question": "Which dataset do the author use in this paper?"
}
```

****

<span id='train_pandagpt'/>

### 3. Train Your Own Mozi Model: <a href='#all_catelogue'>[Back to Top]</a>

**Prerequisites:** Before training the model, making sure the environment is properly installed and the checkpoints of LLaMA-7B and Baichuan-7B are downloaded.

<span id='data_preparation'/>

#### 3.1. Data Preparation: <a href='#all_catelogue'>[Back to Top]</a>

**Declaimer:** To ensure the reproducibility of our results, we have released our training dataset. The dataset must be used for research purpose only. The use of the dataset must comply with the licenses from original sources, i.e. QASPER and SciMRC. These datasets may be taken down when requested by the original authors.

|**Training Task**|**Dataset Address**|
|:-------------:|:-------------:|
|Scientific Pre-training| [Redpajama Dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)|
|Paper-ground Dataset QASPER|[QASPER-v0.3 dataset](https://allenai.org/data/qasper)|
|Paper-ground Dataset SciMRC|[SciMRC dataset]()|
|Emotional Dataset|[scientific-emotional-dialog]()|

Due to the limited computation resources, we only collect 4B tokens from Redpajama arXiv corpus for the first version of scientific pre-training, and the downloading scripts could be found in [this scripts](./scillm/data/pretrain/download_from_hf.py).

After downloading, put the downloaded file and unzip them under the [./data/](./data/) directory.

> **** The directory should look like:

    .
    └── ./data/ 
         ├── pretrain
         └── sft
             ├── qasper
             ├── 000000306060.jpg
             └── ...
              
After downloading these datasets and saving them at the proper path, please refer to [Dataset Prepareing Tutorial](./data/README.md) for preprocessing these for corpus for following training.

<span id='training_configurations'/>

#### 3.2 Training Configurations: <a href='#all_catelogue'>[Back to Top]</a>

To train the model properly, we use the [QLoRA](https://github.com/artidoro/qlora) and [deepspeed](http://deepspeed.readthedocs.io/) toolkit. Before running, please make sure these essential toolkit are downloaded.

The configurations about training are shown as follows:
| Models | Model Name | Training Configurations |
| ------ | ---------- | ----------------------- |
| Scientific Pretraining | scillm | [scillm-train](./config/scillm.json); [scillm-deepspeed](./dsconfig/scillm.json) |
| Paper-ground SFT | scillm-sft | [scillm-sft-train](./config/scillm-sft.json); [scillm-deepspeed](./dsconfig/scillm-sft.json) |
| Paper-ground SFT | scillm-sft | [scillm-sft-train](./config/scillm-sft.json); [scillm-deepspeed](./dsconfig/scillm-sft.json) |

Please refer to these configuration file for more training details. 
Note that these models are pre-trained on 8 x 3090 (24G) GPUs for over 9 days (over 4B tokens). As for paper-ground question answering SFT, the training process cost less than 3 hours (with 2000 steps).


<span id='model_training'/>


#### 3.3. Training Mozi Models: <a href='#all_catelogue'>[Back to Top]</a>

<span id='scientific_pretraining'/>


##### 3.3.1. Scientific Pre-training Mozi Models: <a href='#all_catelogue'>[Back to Top]</a>

To pre-train Mozi on scientific pre-training corpus with 4B tokens, please run the following commands:

```bash
./scripts/train_pretrain.sh
```

The key arguments of the training script are as follows:
* `--model`: The model name listed in `config/base.json`.
* `--model_path`: The checkpoint for large-scale langauge models, `baichuan-inc/baichuan-7B` for baichuan-7B model and `decapoda-research/llama-7b-hf` for LLaMA-7B.
* `--train_data_path`: The path saves the pretraining corpus.
* `--log_path`: The directory that saves the pre-trained log in tensorboard format. This directory will be automatically created.
* `--save_path`: The directory which saves the trained QLoRA delta weights. This directory will be automatically created.

Note that the total training steps can be set in the `total_step` argument at [./config/base.yaml](./config/base.yaml) file. Set this argument carefully to make sure all the tokens will be used during training.



<span id='paper_ground_training'/>


##### 3.3.2. Supversied Fine-tuning Mozi Models: <a href='#all_catelogue'>[Back to Top]</a>

Furthermore, to supervised fine-tune Mozi models on paper-ground question answering corpus, first make sure the `dataset` is set as `QASPERDataset`, and then please run the following commands:
```bash
./scripts/train_sft.sh
```

The key arguments of the training script are as follows:
* `--model`: The model name listed in `config/base.json`.
* `--model_path`: The checkpoint for large-scale langauge models, `baichuan-inc/baichuan-7B` for baichuan-7B model and `decapoda-research/llama-7b-hf` for LLaMA-7B.
* `--delta_model_path`: The LoRA checkpoint weighted pre-traing in Section 3.3.1. The SFT process will continue optimize these LoRA weights for paper-ground question answering task.
* `--train_data_path`: The path saves the pretraining corpus.
* `--log_path`: The directory that saves the pre-trained log in tensorboard format. This directory will be automatically created.
* `--save_path`: The directory which saves the trained QLoRA delta weights. This directory will be automatically created.

Note that the total training steps can be set in the `total_step` argument at [./config/base.yaml](./config/base.yaml) file. Set this argument carefully to make sure all the tokens will be used during training (2000 steps is enough in our hardware settings).

If you want to train the emotional dialog task, just simply replace the dataset with the path of emotional dataset path,and make sure the `dataset_name` in `config/bash.yaml` should also be set as `EmotionalDataset`.



****

<span id='citation'/>

### Citation:

If you found Mozi models useful in your research or applications, please kindly cite using the following BibTeX:
```
...
```

<span id='technialreport'/>


### Technical Report:

You can refer to our technical report for more details, which is saved in [this path](./mozi_technical_report.pdf).

****

<span id='acknowledgments'/>

### Acknowledgments:


This repo benefits from [OpenAlpaca](https://github.com/yxuansu/OpenAlpaca), [PandaGPT](https://panda-gpt.github.io/). Thanks for their wonderful works!

