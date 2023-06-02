## 数据准备

[Redpajama 数据集](https://github.com/togethercomputer/RedPajama-Data)中的ArXiv和CommonCrawl的部分，收集下载约4B左右的token

## 代码准备

目前的代码可能还存在有少量的地方需要调整和修改，此外，需要添加[flash attention](https://github.com/lm-sys/FastChat/blob/4960ca702c66b9adaa65945746dba34f8d2c8ddc/fastchat/train/llama_flash_attn_monkey_patch.py#L110)以加快后续我们优化和训练模型的速度。

训练模型，请直接运行脚本：

```bash
./scripts/train_pretrain.sh
```

### 2023/06/02版本bugs：
* int4 多卡训练 tensor 所在设备不匹配
* flash attention 的适配未经验证
* **NOTE**: transformers/peft/accelerate 库均需 clone github 最新 repo 后安装
    ```commandline
    pip install -q -U git+https://github.com/huggingface/transformers.git
    pip install -q -U git+https://github.com/huggingface/peft.git
    pip install -q -U git+https://github.com/huggingface/accelerate.git
    ```

## Demo 展示页准备

可以参考一些现有的visual-language model的项目，比如[PandaGPT](https://panda-gpt.github.io/), [MiniGPT-4](https://minigpt-4.github.io/), [LLaVa](https://llava-vl.github.io/)
