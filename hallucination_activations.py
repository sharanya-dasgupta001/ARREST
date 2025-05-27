# Pyvene method of getting activations
import os,json
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyvene as pv
from utils import Collector

hf_token = "<INPUT_YOUR_HF_ACCESS_TOKEN>"

# Set the device to GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HuggingFace model identifiers
MODELS_NAMES = {
    'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama3_8B': "meta-llama/Llama-3.1-8B", 
    'vicuna_7B' : "lmsys/vicuna-7b-v1.5" 
}

def load_coqa_dataset():
    """
    Downloads and processes the CoQA dataset.
    Returns:
        Dataset: The processed CoQA dataset.
    """
    import urllib.request
    save_path = './coqa_dataset'
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(f"{save_path}/coqa-dev-v1.0.json"):
        # Download the CoQA dataset if not already present
        url = "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json"
        try:
            urllib.request.urlretrieve(url, f"{save_path}/coqa-dev-v1.0.json")
        except Exception as e:
            print(f"Failed to download coqa dataset file: {e}")
    
    # Load and process the dataset
    with open('./coqa_dataset/coqa-dev-v1.0.json', 'r') as infile:
        data = json.load(infile)['data']
        dataset = {
            'story': [],
            'question': [],
            'answer': [],
            'additional_answers': [],
            'id': []
        }
        for sample in data:
            story = sample['story']
            questions = sample['questions']
            answers = sample['answers']
            additional_answers = sample['additional_answers']
            for question_index, question in enumerate(questions):
                dataset['story'].append(story)
                dataset['question'].append(question['input_text'])
                dataset['answer'].append({
                    'text': answers[question_index]['input_text'],
                    'answer_start': answers[question_index]['span_start']
                })
                dataset['id'].append(sample['id'] + '_' + str(question_index))
                additional_answers_list = [
                    additional_answers[str(i)][question_index]['input_text'] for i in range(3)
                ]
                dataset['additional_answers'].append(additional_answers_list)
                story += f' Q: {question["input_text"]} A: {answers[question_index]["input_text"]}'
                if story[-1] != '.':
                    story += '.'
        return Dataset.from_dict(dataset)
    
def wrapper(intervener):
    """
    Wraps an intervention function to conform to pyvene interface.
    """
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped
    
    
def get_llama_activations_pyvene(collected_model, collectors, prompt, device):
    """
    Computes and collects activations using pyvene from a model given a prompt.
    """
    with torch.no_grad():
        prompt = prompt.to(device)
        output = collected_model({"input_ids": prompt})
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            states_per_gen = torch.stack(collector.states, axis=0).cpu().numpy()
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()
        torch.cuda.empty_cache()
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).squeeze().cpu().numpy()
    return head_wise_hidden_states


def formatter(with_answer, dataset_name, dataset, tokenizer): 
    """
    Formats prompts with or without ground-truth answers for different datasets.
    """
    if with_answer:
        base_prompts = {
        'truthfulqa': "Answer the question concisely. Q: {question} A: {answer}",
        'triviaqa': "Answer the question concisely. Q: {question} A: {answer}",
        'tydiqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A: {answer}",
        'coqa': "Answer the question concisely based on the context: \n {story} \n Q: {question} A: {answer}"

    }
    else :
        base_prompts = {
        'truthfulqa': "Answer the question concisely. Q: {question}",
        'triviaqa': "Answer the question concisely. Q: {question}",
        'tydiqa': "Answer the question concisely based on the context: \n {context} \n Q: {question}",
        'coqa': "Answer the question concisely based on the context: \n {story} \n Q: {question}",
    }
    base_prompt = base_prompts.get(dataset_name, "{prompt}")

    all_prompts = []
        
    for _, row in enumerate(dataset):
        if with_answer:
            if dataset_name in ['truthfulqa']:
                prompt = tokenizer(base_prompt.format(question=row['question'], answer=row['best_answer']), truncation=True, return_tensors='pt').input_ids
            if dataset_name in ['triviaqa']:
                prompt = tokenizer(base_prompt.format(question=row['question'], answer=row['answer']), truncation=True, return_tensors='pt').input_ids
            elif dataset_name == 'tydiqa':
                prompt = tokenizer(base_prompt.format(context=row['context'], question=row['question'], answer=row['answers']), truncation=True,  return_tensors='pt').input_ids
            elif dataset_name == 'coqa':
                prompt = tokenizer(base_prompt.format(story=row['story'], question=row['question'], answer=row['answers']), truncation=True,  return_tensors='pt').input_ids
        else :
            if dataset_name in ['truthfulqa']:
                prompt = tokenizer(base_prompt.format(question=row['question']), return_tensors='pt').input_ids
            if dataset_name in ['triviaqa']:
                prompt = tokenizer(base_prompt.format(question=row['question']), return_tensors='pt').input_ids
            elif dataset_name == 'tydiqa':
                prompt = tokenizer(base_prompt.format(context=row['context'], question=row['question']), return_tensors='pt').input_ids
            elif dataset_name == 'coqa':
                prompt = tokenizer(base_prompt.format(story=row['story'], question=row['question']), return_tensors='pt').input_ids
            
        all_prompts.append(prompt)
        
    return all_prompts

def main(): 
    """
    Main script for extracting and saving head-wise LLM hidden state activations
    using pyvene from a specified dataset and model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', choices=MODELS_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='truthfulqa', choices=['truthfulqa', 'triviaqa', 'tydiqa', 'coqa'], help='Dataset')
    parser.add_argument('--with_answer', action='store_true', help='Include answer context')
    
    args = parser.parse_args()

    MODEL = MODELS_NAMES[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        use_auth_token=hf_token,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        cache_dir="./models",
        attn_implementation="eager").to(DEVICE)

    # Load the appropriate dataset
    if args.dataset_name == "truthfulqa":
        dataset = load_dataset("truthful_qa", 'generation')['validation']
    elif args.dataset_name == 'triviaqa':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()
        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch
        dataset = dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
        dataset = dataset.map(lambda example: {'answer': example['answer']['value']})
    elif args.dataset_name == 'tydiqa':
        dataset = load_dataset("tydiqa", "secondary_task", split="train")
        dataset = dataset.filter(lambda row: "english" in row["id"])
        dataset = dataset.map(lambda example: {'answers': example['answers']['text'][0]})
    elif args.dataset_name == 'coqa':
        dataset =  load_coqa_dataset()
        dataset = dataset.map(lambda example: {'answers': example['answer']['text']})
    else: 
        raise ValueError("Invalid dataset name")

    # Prepare prompts
    print("\nTokenizing prompts\n")
    prompts = formatter(args.with_answer, args.dataset_name, dataset, tokenizer)
    
    # Setup collectors and pyvene configuration for each transformer layer
    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers): 
        collector = Collector(multiplier=0, head=-1) 
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })
    collected_model = pv.IntervenableModel(pv_config, model)
    all_head_wise_activations = []
    tqdm.write('')
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc=f"Getting activations for {args.dataset_name} dataset ..."):
        try :
            head_wise_activations = get_llama_activations_pyvene(collected_model, collectors, prompt, DEVICE)
            all_head_wise_activations.append(head_wise_activations.copy())
        except:
            all_head_wise_activations.append(np.full_like(all_head_wise_activations[-1], -1))
            print(f"Out of memory error, skipping {i}")
        torch.cuda.empty_cache()
    
    # Save activations
    os.makedirs(f'./hallucination/hidden', exist_ok=True)
    print("\nSaving head wise activations")
    if args.with_answer:
        np.save(f'./hallucination/hidden/{args.model_name}_{args.dataset_name}_head_wise_truth.npy', all_head_wise_activations)
    else:
        np.save(f'./hallucination/hidden/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)
    print("\n__________________________________________________________________________________________\n")

if __name__ == '__main__':
    main()