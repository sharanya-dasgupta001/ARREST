# 🚨 ARREST: Adversarial Resilient Regulation Enhancing Safety and Truth in Large Language Models

Welcome to the official repository of **ARREST** 🧠 — a framework for improving **truthfulness** and **safety** in Large Language Models (LLMs) via *intervention-based adversarial training*. 

---

## 📜 Abstract

> **ARREST** introduces an *external adversarial network* trained to monitor and selectively intervene during inference, regulating:
- ❌ **Hallucinations → Truthful output**
- ⚠️ **Unsafe generations → Safe responses**

**ARREST** supports :
- 🤝 Soft refusals
- 🛑 Hard denials
- ✅ Truthfulness restoration

🧪 Our results demonstrate ARREST’s superiority over RLHF-aligned models in generating contextually nuanced refusals and improving factual accuracy through adversarial interventions.

---

## 🧭 Table of Contents
1. [⚙️ Setup](#-setup)
2. [🧠 Hallucination Evaluation](#-hallucination-evaluation)
3. [🛡️ Safety Evaluation](#-safety-evaluation)
4. [📝 Notes](#-note)

---

## ⚙️ Setup

In the root folder of this repo, run the following commands to set things up.
- Install `Python 3.10.12` and the necessary packages from `requirements.txt`.
- For easily managing different python versions, we recommend using [conda](https://docs.anaconda.com/miniconda/install/).
- Create a new environment in conda 🐍 and install necessary python packages:
    ```bash
    conda create -n arrest python=3.10.12 -y
    conda activate arrest
    pip install -r requirements.txt
     ```
- 📁 Directory Setup :
    ```bash
   mkdir models
   mkdir -p hallucination/hidden
   mkdir -p hallucination/responses
   mkdir -p safety/hidden
   mkdir -p safety/responses
   ```
    
- 🔐 HuggingFace Access Token :
   - Login to `huggingface` or create an account if you don't have already.
   - From the [settings](https://huggingface.co/settings/tokens) create a new access token with WRITE access.
   - Open the the files and paste your access token at beginning `hf_token = "<INPUT_YOUR_HF_ACCESS_TOKEN>"`

---

 ## 🧠 Hallucination Evaluation

1. 🔍 Get Activations :
    ```bash
    model=llama2_7B dataset=truthfulqa bash get_activations_hal.sh
    ```
    - `model`: Choose from [`llama2_7B`](https://huggingface.co/meta-llama/Llama-2-7b-hf), [`llama3_8B`](https://huggingface.co/meta-llama/Llama-3.1-8B), or [`vicuna_7B`](https://huggingface.co/lmsys/vicuna-7b-v1.5).
    - `dataset`: Choose from [`truthfulqa`](https://huggingface.co/datasets/truthfulqa/truthful_qa/viewer/generation/validation), [`triviaqa`](https://huggingface.co/datasets/mandarjoshi/trivia_qa), [`tydiqa`](https://huggingface.co/datasets/google-research-datasets/tydiqa), [`coqa`](https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json).

2. 🧪 Train and Infer ARREST-Adversarial-Hallucination : 
    ```bash
   python hallucination_adversarial.py --model_name llama2_7B  --dataset_name truthfulqa --num_layers 1 --num_fold 5
    ``` 
   - `model_name`: Choose from [`llama2_7B`](https://huggingface.co/meta-llama/Llama-2-7b-hf), [`llama3_8B`](https://huggingface.co/meta-llama/Llama-3.1-8B), or [`vicuna_7B`](https://huggingface.co/lmsys/vicuna-7b-v1.5).
   - `dataset_name`: Choose from [`truthfulqa`](https://huggingface.co/datasets/truthfulqa/truthful_qa/viewer/generation/validation), [`triviaqa`](https://huggingface.co/datasets/mandarjoshi/trivia_qa), [`tydiqa`](https://huggingface.co/datasets/google-research-datasets/tydiqa), [`coqa`](https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json).
   - Truthfulness (%) will be printed on screen and responses will be saved into ```hallucination/responses``` folder.

## 🛡️ Safety Evaluation

1. 🔍 Get Activations :
    ```bash
    model=llama2_7B dataset=malicious-instruct bash get_activations_safety.sh
    ```
    -  `model`: Choose from [`llama2_7B`](https://huggingface.co/meta-llama/Llama-2-7b-hf), [`llama3_8B`](https://huggingface.co/meta-llama/Llama-3.1-8B), [`Qwen2.5_7B`](https://huggingface.co/Qwen/Qwen2.5-7B), or [`Yi1.5_9B`](https://huggingface.co/01-ai/Yi-1.5-9B).
    - `dataset`: Choose from [`malicious-instruct`](https://huggingface.co/datasets/walledai/MaliciousInstruct), [`advbench`](https://huggingface.co/datasets/walledai/AdvBench), [`jailbreak-bench`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), [`trustllm`](https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset).

2. ⚔️ Train and Infer ARREST-Adversarial-Safety : 
    ```bash
   python safety_adversarial.py --model_name llama2_7B  --dataset_name malicious-instruct --num_layers 1 --num_fold 5
    ``` 
    - `model_name`: Choose from [`llama2_7B`](https://huggingface.co/meta-llama/Llama-2-7b-hf), [`llama3_8B`](https://huggingface.co/meta-llama/Llama-3.1-8B), [`Qwen2.5_7B`](https://huggingface.co/Qwen/Qwen2.5-7B), or [`Yi1.5_9B`](https://huggingface.co/01-ai/Yi-1.5-9B).
    - `dataset_name`: Choose from [`malicious-instruct`](https://huggingface.co/datasets/walledai/MaliciousInstruct), [`advbench`](https://huggingface.co/datasets/walledai/AdvBench), [`jailbreak-bench`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), [`trustllm`](https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset).
    - Attack Success Rate(%) will be printed on screen and responses will be saved into ```safety/responses``` folder.

3. 🧲 Train and Infer ARREST-Contrastive-Safety : 
    ```bash
   python safety_contrastive.py --model_name llama2_7B  --dataset_name malicious-instruct --num_layers 1 --num_fold 5
    ``` 
    - `model_name`: Choose from [`llama2_7B`](https://huggingface.co/meta-llama/Llama-2-7b-hf), [`llama3_8B`](https://huggingface.co/meta-llama/Llama-3.1-8B), [`Qwen2.5_7B`](https://huggingface.co/Qwen/Qwen2.5-7B), or [`Yi1.5_9B`](https://huggingface.co/01-ai/Yi-1.5-9B).
    - `dataset_name`: Choose from [`malicious-instruct`](https://huggingface.co/datasets/walledai/MaliciousInstruct), [`advbench`](https://huggingface.co/datasets/walledai/AdvBench), [`jailbreak-bench`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), [`trustllm`](https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset).
    - Attack Success Rate(%) will be printed on screen and responses will be saved into ```safety/responses``` folder.
---

### 📝 Notes
1. ✅ **Ground Truth Evaluation**  
   To evaluate generated answers with ground truth (for hallucination), we use [BleuRT](https://arxiv.org/abs/2004.04696) to evaluate truthfulness.
   - To install BleuRT run:
   ```bash
   pip install --upgrade pip  # ensures that pip is current
   git clone https://github.com/google-research/bleurt.git
   cd bleurt
   pip install .
   ```
   - 💡 using 12-layer distilled model for faster inference, which is ~3.5X smaller.
   - Download the model and save it in the `./models` folder:
   ```
   wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip
   unzip BLEURT-20-D12.zip
   mv BLEURT-20-D12 models/. # Move the bleurt model folder to models directory
   ```
   - If you want to use any different model please refer to [BleuRT repository](https://github.com/google-research/bleurt).

2. 🧬 **Intervention**  
[pyvene](https://github.com/stanfordnlp/pyvene) is really cool library that can be used to load Inference-time Intervention ⚙️ , and many other mechanistic intervention 🧩 technique.
---