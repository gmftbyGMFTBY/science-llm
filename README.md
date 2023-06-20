# Recoding PPL


| Checkpoints        | PPL       |
| ------------------ | --------- |
| LLaMA-7B           | 72.5932   |
| SciLLM-7B (5%)     | 3.7634    |
| SciLLM-7B (10%)    | 3.6565    |
| SciLLM-7B (15%)    | 3.6122    |
| SciLLM-7B (20%)    | 3.5838    |
| SciLLM-7B (25%)    | 3.5719    |
| SciLLM-7B (30%)    | 3.5711    |
| SciLLM-7B (35%)    | 3.5704    |
| SciLLM-7B (40%)    | 3.5612    |
| SciLLM-7B (55%)    | 3.5341    |


| Checkpoints        | PPL       |
| ------------------ | --------- |
| Baichuan-7B        | 6.4137    |
| SciLLM-7B (15%)    | 3.0814    |


# QASPER Zero-shot Evaluation


| Checkpoint | Acc |
| ---------- | --- |
| LLaMA-7B   | 0.2568 |
| ChatGPT    |  |
| SciLLM-7B (45%) | 0.2643 |
| SciLLM-7B (55%) | 0.2634 |

# Generation Evaluation

## Generation Evaluation on QASPER test set

### Baselines

7B models are evaluated:
1. Alpaca (tianyi)
2. Vicuna (tianyi)
3. OpenAlpaca (tianyi)
6. BaiChuan (jinyu)
4. dolly (shuhang)
5. ChatGLM (shuhang)
7. LongFormer baseline in QASPER (shuhang)

### Evaluation Metrics

1. BLEU/ROUGE: 因为生成内容和答案一般邀请事实准确度，所以我们希望可以尽可能地匹配上
2. BERTScore
3. ChatGPT evaluation: 定义好对应地prompt，要求ChatGPT重点考虑和参考答案地匹配接近程度

### Setting

1. Greedy Search
2. 生成最大长度为128或者到eos token提前结束生成
3. 最大上下文2048长度
4. Prompt需要包含以下两点内容：
    * evidence：字符串，多个evidence使用\n拼接
    * question: 用户的问题


# TODO List

- [ ] train the SciMRC and QASPER on Baichuan-7B model
- [x] train the Emptional Scientific Dialog on Baichuan-7B model
- [x] make the api deployment
- [ ] evaluate the performance between our model and other strong SFT baselines
- [ ] evaluate the performance of our scientific pre-training
- [ ] write the README
- [ ] write the technical report
- [ ] write the demo page
