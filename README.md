![Alpaca-CoT](https://github.com/PhoebusSi/alpaca-CoT/blob/main/figures/Alpaca-CoT-2.jpg)
# Evolving Alpaca: An Empirical Study on Instruction Tuning for Large Language Models (**Alpaca-CoT**)

中文README，请看[这里](https://github.com/PhoebusSi/Alpaca-CoT/blob/main/CN_README.md)。(Chinese READEME can be found [here](https://github.com/PhoebusSi/Alpaca-CoT/blob/main/CN_README.md).)

This is the repository for the `Evolving Alpaca` project, which aims to extensively collect instruction-tuning datasets (especially the CoT datasets) and conduct an in-depth empirical study based on [LLaMA](https://arxiv.org/abs/2302.13971v1) model [1].  `Evolving` is used to describe the continuous expansion of our [instruction-tuning data collection](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/), which will continuously enhance [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)'s [2] instruction-following capabilities.

You are in a warm welcome to provide us with any non-collected instruction-tuning datasets (or their sources). We will uniformly format them, train Alpaca model (and other LLMs in the early future) with these datasets, open source the [model checkpoints](https://huggingface.co/QingyiSi/Alpaca-CoT/tree/main), and conduct extensive empirical studies. We hope that our project can make a modest contribution to the open-source process of large language models, and reduce its threshold for NLP researchers to get started.

## News
- 3.30: LLM ChatGLM (THUDM/chatglm-6b) is merged in `uniform_finetune.py`.
- 3.29: LLM BLOOM (bloomz-7b1-mt) is merged in `uniform_finetune.py`. (The corresponding model and command will be released later.)
- 3.28: To facilitate downloading, all model(LoRA) weights have been uploaded [here](https://huggingface.co/QingyiSi/Alpaca-CoT/tree/main).
- 3.28: Chinese instruction dataset(1M, Does not contain the original 0.5M ones) published by BELLE has been formatted and collected [here](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main). (The model will be released later.)

## Overview

LLaMA [1] is a great work that demonstrates the amazing zero-shot and few-shot ability. It significantly reduces the cost of training, finetuning, and using competitive large language models, i.e., LLaMA-13B outperforms GPT-3(175B) and LLaMA-65B is competitive to PaLM-540M. Recently, to boost the instruction-following ability of LLaMA, Stanford Alpaca [2] finetuned LLaMA-7B on 52K instruction-following data generated by the [Self-Instruct](https://arxiv.org/abs/2212.10560) [3] techniques. However, at present, the LLM research community still faces two challenges: 1. Even LLaMA still has high requirements for computing resources, and 2. There are not many open source datasets for instruction finetuning. 

To this end, we propose this project, which leverages various improvements that were subsequently proposed, with the following advantages:
- This repo contains code, modified from [here](https://github.com/tloen/alpaca-lora), which can **_finetune LLaMA cheaply and efficiently_** (without performance degradation compared to Stanford Alpaca) by using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf) [4], [PEFT](https://github.com/huggingface/peft) and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). The `7b`, `13b` and `30b` versions of LLaMA models can be easily trained on a single 80G A100. 
- The models published in this repo significantly **_improve the CoT (reasoning) capability_**, using CoT datasets published by FLAN [5].
- The models published in this repo significantly **_improve the ability to follow Chinese instructions_**, with the help of Chinese instruction datasets published by BELLE [6].
- This repo contains **_a [collection of instruction-finetuning datasets](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT) that are continuously collected_**, which so far includes English, Chinese and CoT instructions. In addition, a collection of checkpoints trained with various instruction datasets is also provided.
- This repo contains **_extensive empirical studies and qualitative analysis_**, which may provide valuable findings and promote the exploration of LLM in the future.


**To the best of our knowledge, this work is the first to study _CoT reasoning_ based on LLaMA and Alpaca.** Therefore, we abbreviate our work to `Alpaca-CoT`.


[1]: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971v1)

[2]: [Stanford Alpaca: An Instruction-following LLaMA model](https://github.com/tatsu-lab/stanford_alpaca)

[3]: [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560)

[4]: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)

[5]: [FLAN: Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)

[6]: [BELLE: Bloom-Enhanced Large Language model Engine](https://github.com/LianjiaTech/BELLE)

## Data Collection 
### Statistics
![data collection statistics](https://github.com/PhoebusSi/alpaca-CoT/blob/main/figures/piechart.png)
The current collection of instruction-finetuning datasets consists mainly of three parts:
- `alpaca_data_cleaned.json`: about 52K English instruction-following training samples.
- `belle_data_cn.json`:  about 0.5M Chinese |instruction-following training samples. 
- `CoT_data.json`: 9 CoT datasets involving about 75k samples.

More details on the usage and sources of different datasets can be found [here](https://github.com/PhoebusSi/alpaca-CoT/tree/main/data). 
### Download
You can download all the formatted data [here](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main). Then you should put them in the [data](https://github.com/PhoebusSi/alpaca-CoT/tree/main/data) folder. 

You can download all checkpoints trained on various types of instruction data from [here](https://huggingface.co/QingyiSi/Alpaca-CoT/tree/main). Then, after setting `LoRA_Weights` (in `generate.py`) to the local path, you can directly execute the model inference. 


### Data Fomatting
All data in our collection is formatted into the same templates, where each sample is as follows:
```
[
{"instruction": instruction string,
"input": input string, # (may be empty)
"output": output string}
]
```
Note that, for CoT datasets, we first use the [template](https://github.com/google-research/FLAN/blob/main/flan/v2/templates.py) provided by FLAN to change the original dataset into various Chain-of-Thoughts forms, and then convert it to the above format. The formatting script can be found [here](https://github.com/PhoebusSi/alpaca-CoT/blob/main/data/origin_cot_data/formating.py).

## Dict of Data_type and Path

```
             "alpaca": "./data/alpaca_data_cleaned.json",
             "belle": "./data/belle_data_cn.json",
             "alpaca-belle": "./data/alpaca_plus_belle_data.json",
             "cot": "./data/CoT_data.json",
             "alpaca-cot": "./data/alcapa_plus_cot.json",
             "alpaca-belle-cot": "./data/alcapa_plus_belle_plus_cot.json",
             "belle1.5m": "./data/belle_data1.5M_cn.json",
             "finance": "./data/finance_en.json"
```

## Instruction Finetuning
### Setup
```
pip install -r requirements.txt
```
Note that, make sure python>=3.9 when finetuning ChatGLM.
### Instruction Tuning

**Single GPU**
- for LLaMA
```
python3 uniform_finetune.py --model_type llama --model_name_or_path decapoda-research/llama-7b-hf \
    --data alpaca-belle-cot --lora_target_modules q_proj v_proj \
    --per_gpu_train_batch_size 128 --learning_rate 3e-4 --epochs 1 
    
```
- for ChatGLM
```
python3 uniform_finetune.py   --model_type chatglm --model_name_or_path THUDM/chatglm-6b \
    --data alpaca-belle-cot --lora_target_modules query_key_value \
    --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 --per_gpu_train_batch_size 2 \
    --learning_rate 2e-5 --epochs 1
```
Note that `load_in_8bit` is not yet suitable for ChatGLM, so batch_size must be much smaller than others. 

- for BLOOM
```
python3 uniform_finetune.py   --model_type bloom --model_name_or_path bigscience/bloomz-7b1-mt \
    --data alpaca-belle-cot --lora_target_modules query_key_value \
    --per_gpu_train_batch_size 128 --learning_rate 3e-4 --epochs 1 
```

Note that you can also pass the local path (where the LLM weights saved) to `--model_name_or_path`. And the data type `--data` can be freely set according to your interests.

**Multiple GPUs**
- for LLaMA
```
python3 -m torch.distributed.launch --nproc_per_node 4  \
    --nnodes=1 --node_rank=0 --master_addr=xxx --master_port=yyy uniform_finetune.py \
    --model_type llama --model_name_or_path decapoda-research/llama-7b-hf \
    --data alpaca-belle-cot --lora_target_modules q_proj v_proj \
    --per_gpu_train_batch_size 4 --learning_rate 3e-4 --epochs 1 
```
- for ChatGLM
```
python3 -m torch.distributed.launch --nproc_per_node 4  \
    --nnodes=1 --node_rank=0 --master_addr=xxx --master_port=yyy \
    uniform_finetune.py   --model_type chatglm --model_name_or_path THUDM/chatglm-6b \
    --data alpaca-belle-cot --lora_target_modules query_key_value \
    --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 --per_gpu_train_batch_size 2 \
    --learning_rate 2e-5 --epochs 1
```
Note that `load_in_8bit` is not yet suitable for ChatGLM, so batch_size must be smaller than others. 

- for BLOOM
```
python3 -m torch.distributed.launch --nproc_per_node 4  \
    --nnodes=1 --node_rank=0 --master_addr=xxx --master_port=yyy \
    uniform_finetune.py   --model_type bloom --model_name_or_path bigscience/bloomz-7b1-mt \
    --data alpaca-belle-cot --lora_target_modules query_key_value \
    --per_gpu_train_batch_size 4 --learning_rate 3e-4 --epochs 1  
```

### Inference #To be modified
``` 
python3 generate.py --size 7 --data alpaca-belle-cot

```
More details of instruction finetuing and inference can be found [here](https://github.com/tloen/alpaca-lora) where we modified from. Note that the folders `saved-xxx7b` are the save path for LoRA weights, and LLaMA weights are automatically downloaded from Hugging Face.

## Quantitative Analysis

### Ablation of CoT and Chinese Instructions


![ablation-cot](https://github.com/PhoebusSi/alpaca-CoT/blob/main/figures/ablation-cot.png)
"w/o CoT" and "w/o CN" denote models that exclude CoT data and Chinese instructions from their instruction finetuning data, respectively.  

The above table shows two examples (invoving with numerical calculations) that require a certain amount of reasoning ability to respond correctly. 
As shown in the middle column, `Ours w/o CoT` fails to generate the correct response, which shows that once the finetuning data does not contain CoT data, the model's reasoning ability significantly decreases. This further demonstrates that CoT data is essential for LLM models. 

![ablation-cot](https://github.com/PhoebusSi/alpaca-CoT/blob/main/figures/ablation-cn.png)

The above table shows two examples that require the ability to respond to Chinese instructions. 
As shown in the right column, either the generated content of `Ours w/o CN` is unreasonable, or the Chinese instructions are answered in English by `Ours w/o CN`. This shows that removing Chinese data during finetuning will cause the model to be unable to handle Chinese instructions, and further demonstrates the need to collect Chinese instruction finetuning data.


![ablation-cot](https://github.com/PhoebusSi/alpaca-CoT/blob/main/figures/ablation-both.png)

The above table shows a relatively difficult example, which requires both a certain accumulation of knowledge of Chinese history and a logical and complete ability to state historical events. As shown in this table, `Ours w/o CN` can only generate a short and erroneous response, because due to the lack of Chinese finetuning data, the corresponding knowledge of Chinese history is naturally lacking.  Although `Ours w/o CoT` lists some relevant Chinese historical events, its logic of expression is self-contradictory, which is caused by the lack of CoT data.
`

**In summary, the models finetuned from our complete dataset (English, Chinese, and CoT instruction data) can significantly improve model reasoning and Chinese instruction following abilities.**

### The Effect of CoT Data
  
![CoT-comparison](https://github.com/PhoebusSi/alpaca-CoT/blob/main/figures/CoT-comparison.png)
Samples of each odd number of rows do not apply the CoT prompt, such as "step-by-step reasoning." Both `Ours(w/CoT)` and Alpaca are based on LLaMA-7B, and the only difference between them two is that the instruction-finetuning data of `Ours(w/CoT)` has a extra CoT data than that of Alpaca. 

From the above table, we find that:
- `Ours(w/CoT)` always generates the correct rationale before the answer, while Alpaca fails to generate any reasonable rationale, as shown in the first 4 examples (commonsense questions). This shows that using CoT data for finetuning can significantly improve reasoning ability. 
- For `Ours(w/CoT)`, the CoT prompt (e.g., concatenate 'step-by-step' with the input question) has little effect on easy examples (e.g., commonsense questions) and has an important effect on challenging questions (e.g., questions requiring reasoning, like the last four examples). 
- For Alpaca, CoT prompt always has little effect or even negative impact. For the last two examples, after adding CoT prompt, Aplpaca changes the correct generated answer to the wrong one. This may be due to the inconsistency between the input forms of finetuning and inference. 


### The Effect of Chinese Instruction Data

_Quantitative comparison of responses to Chinese instructions._
![CN_compare_CN](https://github.com/PhoebusSi/alpaca-CoT/blob/main/figures/CN-compareCN.png)

Our model is finetuned from a 7B LLaMA on 52K English instructions and 0.5M Chinese instructions. Stanford Alpaca (our reimplementation) is finetuned from a 7B LLaMA on 52K English instructions. BELLE is finetuned from a 7B BLOOM on 2B Chinese instructions. 

From the above table, several observations can be found:
- Compared to Alpaca, `ours (w/ CN)` has a stronger ability to understand Chinese instructions. For the first example, Alpaca fails to distinguish between the `instruction` part and `input` part, while we do.
- Chinese instruction finetuning data can significant enhance the ability to interact in Chinese. For the second example, `ours (w/ CN)` not only provides the correct code, but also provides the corresponding Chinese annotation, while Alpaca does not. In addition, as shown in the 3-5 examples, Alpaca can only respond to Chinese instruction with an English response. 
- Compared to BELLE, `ours (w/ CN)`'s performance on instructions requiring an open response (as shown in last two examples) still needs to be improved. BELLE's outstanding performance against such instructions is due to: 1. Its BLOOM backbone model encounters much more multilingual data during pre-training; 2. Its Chinese instruction finetuning data is more than ours, that is, 2M vs 0.5M.



 _Quantitative comparison of responses to English instructions. The purpose of this subsection is to explore whether finetuning on Chinese instructions has a negative impact on Alpaca._ 
![CN_compare_EN](https://github.com/PhoebusSi/alpaca-CoT/blob/main/figures/CN_compareEN.png)


From the above table, we find that: 
- Finetuning with Chinese instruction data does not weaken the original English instruction–following ability, on the contrary, there is also a certain enhancement in genearting a better response to English intructions. The response of `ours (w/ CN)` shows more detail than that of Alpaca, e.g. for the third example, `ours (w/ CN)` list three more provinces than Alpaca.  





## Future Work
- Exploration of few-shot ability.
- Ablation study of various sizes of models.
- Evaluate on instruction-following evaluation suite.
- Collect more instruction finetuning datasets.

  
## Citation
Please cite the repo if you use the data collection, code, and experimental findings in this repo. 
```
@misc{alpaca-cot,
  author = {Qingyi Si, Rui Liu, Zheng Lin },
  school = {Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China},
  title = {Evolving Alpaca: An Empirical Study on Instruction Tuning for Large Language Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PhoebusSi/alpaca-CoT}},
}
```
For data, please cite the original Stanford Alpaca, BELLE and FLAN papers as well.

For models, please cite the original LLaMA, Stanford Alpaca, Self-Instruct and LoRA papers as well.
