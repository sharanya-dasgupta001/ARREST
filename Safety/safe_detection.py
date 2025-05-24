import functions, classifier
from datasets import load_dataset, Dataset
import json
import time
import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from concurrent.futures import ThreadPoolExecutor
import random

# Add your Hugging Face Access Token here
hf_token = "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"

# Suppress warnings
warnings.filterwarnings("ignore")

# Mapping of model names to their identifiers
MODELS_NAMES = {
    'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama3_8B': "meta-llama/Llama-3.1-8B",
    'opt6.7B': "facebook/opt-6.7b"
}

def seed_everything(seed: int):
    """Sets seeds for reproducibility across various libraries.
    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_dataset_by_name(dataset_name):
    """Loads a dataset based on the provided name.
    Args:
        dataset_name (str): The name of the dataset to load.
    Returns:
        Dataset: The loaded dataset.
    """
    
    # Load the TruthfulQA dataset's validation split
    if dataset_name == "truthfulqa":
        return load_dataset("truthful_qa", 'generation')['validation']
    
    # Load the TriviaQA dataset and remove duplicate questions
    elif dataset_name == 'triviaqa':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()
        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch
        return dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    
    # Load the TyDiQA dataset and filter for English questions
    elif dataset_name == 'tydiqa':
        dataset = load_dataset("tydiqa", "secondary_task", split="train")
        return dataset.filter(lambda row: "english" in row["id"])
    
    # Load the CoQA dataset
    elif dataset_name == 'coqa':
        return load_coqa_dataset()
    
    # Load a specific subset of the HaluEval dataset
    elif dataset_name == 'haluevaldia':
        return load_dataset("pminervini/HaluEval", "dialogue")['data']
    elif dataset_name == 'haluevalqa':
        return load_dataset("pminervini/HaluEval", "qa")['data']
    elif dataset_name == 'haluevalsum':
        return load_dataset("pminervini/HaluEval", "summarization")['data']
    elif dataset_name == 'combined':
        # Load both datasets
        trustllm_dataset = load_trust_dataset()
        overrefusal_dataset = load_over_refusal_dataset()
        
        # Combine datasets
        dataset = load_combined_dataset(trustllm_dataset, overrefusal_dataset)
        # formatter = tokenized_combined_dataset
        return dataset
    else:
        raise ValueError("Invalid dataset name")

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
    
############ Safety Dataset Loading Functions ############
def load_trust_dataset():
    """
    Downloads and processes the TrustLLM dataset.
    Returns:
        Dataset: The processed TrustLLM dataset.
    """
    save_path = './llm_trust_dataset'
    os.makedirs(save_path, exist_ok=True)
    # if not os.path.exists(f"llm_trust_dataset/misuse.json"):
    #     try:
    #         file_path = hf_hub_download(repo_id="TrustLLM/TrustLLM-dataset", filename="safety/misuse.json", repo_type="dataset")
    #         shutil.copy(file_path, os.path.join(os.getcwd(), "llm_trust_dataset/misuse.json"))
    #     except Exception as e:
    #         print(f"Failed to download TrustLLM dataset file: {e}")
    
    with open("llm_trust_dataset/misuse.json", "r") as file:
        data = json.load(file)  
        return Dataset.from_list(data)

def load_over_refusal_dataset(num_samples=1000):
    """
    Loads the over-refusal dataset and samples a subset.
    Args:
        num_samples: Number of samples to take from the dataset
    Returns:
        Dataset: The sampled over-refusal dataset
    """
    print("Loading over-refusal dataset...")
    dataset = load_dataset("bench-llm/or-bench", "or-bench-80k")['train']
    
    # Randomly sample num_samples from the dataset
    total_samples = len(dataset)
    if num_samples < total_samples:
        indices = random.sample(range(total_samples), num_samples)
        dataset = dataset.select(indices)
    
    print(f"Sampled {len(dataset)} examples from over-refusal dataset")
    return dataset
    
    
def load_combined_dataset(trustllm_dataset, overrefusal_dataset):
    """
    Combines TrustLLM and over-refusal datasets with appropriate labels.
    Args:
        trustllm_dataset: The TrustLLM dataset (unsafe prompts)
        overrefusal_dataset: The over-refusal dataset (safe prompts)
    Returns:
        Dataset: The combined dataset with labels
    """
    # Convert TrustLLM dataset to list format with label 1 (unsafe)
    trustllm_data = [{"prompt": item["prompt"], "label": 1, "type": item.get("type", "harmful")} 
                     for item in trustllm_dataset]
    
    # Convert over-refusal dataset to list format with label 0 (safe)
    overrefusal_data = [{"prompt": item["prompt"], "label": 0, "type": "safe"} 
                       for item in overrefusal_dataset]
    
    # Combine the datasets
    combined_data = trustllm_data + overrefusal_data
    
    # Shuffle the combined dataset
    random.shuffle(combined_data)
    
    return Dataset.from_list(combined_data)

def process_with_threads(args, dataset, process_func, max_workers):
    """Processes a dataset in parallel using threading.
    Args:
        dataset (Dataset): The dataset to process.
        process_func (callable): The function to apply to each dataset entry.
        max_workers (int): The maximum number of threads to use.
    Returns:
        list: Processed dataset entries.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(tqdm(executor.map(process_func, dataset), total=len(dataset), desc=f"Generating responses for {args.dataset_name} dataset ..."))

def main():
    """
    Main function to perform the following tasks:
    - Parse command-line arguments.
    - Download and preprocess datasets.
    - Load and configure language models.
    - Generate responses using the language model.
    - Evaluate generated responses using BLEURT.
    - Train classifier.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2-7B', help='Name of the model to use.')
    parser.add_argument('--dataset_name', type=str, default='truthfulqa', help='Name of the dataset to use.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpu threads to use.')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with limited samples.')
    parser.add_argument('--test_samples', type=int, default=100, help='Number of samples to use in test mode.')
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed_everything(42)

    # Determine model path (local or remote)
    if args.model_name not in MODELS_NAMES.keys():
        print(args.model_name)
        raise ValueError("Invalid model name")
    MODEL = MODELS_NAMES[args.model_name]
    
    print(f"""
    =========================================================================
                        HalluShift Execution Started
    =========================================================================
    Dataset: {args.dataset_name}    Model: {args.model_name}
    {'TEST MODE: Using only ' + str(args.test_samples) + ' samples' if args.test_mode else 'FULL MODE: Using complete dataset'}

    Workflow:
    1. LLM Response Generation and feature collection :
    - Estimated Time: Varies by dataset size, context length, and hardware
    - Example: TruthfulQA on llama2-7B takes ~45-60 min on NVIDIA GeForce RTX 3090

    2. Ground Truth Evaluation :
    - Method: BleuRT evaluation of LLM generated responses
    - Estimated Time: Varies by dataset size and answer length
    - Example: TruthfulQA takes ~30-60 sec on NVIDIA GeForce RTX 3090

    3. Feature Processing and Classifier Training :
    - Estimated Time: Varies by dataset size
    - Example: TruthfulQA takes ~60-90 sec on NVIDIA GeForce RTX 3090

    Output: 
    - Various evaluation metrics are displayed on the screen
    - LLM responses and processed dataset stored in 'result' folder
    =========================================================================\n
    """)
    time.sleep(10)
    
    print("Downloading Dataset...\n")
    dataset = load_dataset_by_name(args.dataset_name)
    print("Dataset successfully downloaded.\n")

    # Limit dataset size if in test mode
    if args.test_mode:
        if args.dataset_name == 'combined':
            # For combined dataset, ensure balanced sampling from both classes
            safe_indices = [i for i, label in enumerate(dataset['label']) if label == 0]
            unsafe_indices = [i for i, label in enumerate(dataset['label']) if label == 1]
            
            # Calculate how many samples to take from each class
            samples_per_class = args.test_samples // 2
            
            # Ensure we have enough samples from each class
            if len(safe_indices) < samples_per_class or len(unsafe_indices) < samples_per_class:
                print(f"Warning: Not enough samples in one of the classes. Found {len(safe_indices)} safe and {len(unsafe_indices)} unsafe examples.")
                samples_per_class = min(len(safe_indices), len(unsafe_indices))
                print(f"Using {samples_per_class * 2} total samples ({samples_per_class} from each class)")
            
            # Sample from each class
            sampled_safe = random.sample(safe_indices, samples_per_class)
            sampled_unsafe = random.sample(unsafe_indices, samples_per_class)
            
            # Combine indices and create balanced dataset
            balanced_indices = sampled_safe + sampled_unsafe
            random.shuffle(balanced_indices)  # Shuffle to avoid having all samples of one class together
            
            dataset = dataset.select(balanced_indices)
            print(f"Test mode: Created balanced dataset with {len(dataset)} examples ({samples_per_class} safe, {samples_per_class} unsafe)")
        elif len(dataset) > args.test_samples:
            # For other datasets, use random sampling
            indices = random.sample(range(len(dataset)), args.test_samples)
            dataset = dataset.select(indices)
            print(f"Test mode: Sampled {args.test_samples} examples from dataset")
        else:
            print(f"Test mode: Dataset has only {len(dataset)} examples, using all of them")

    print("Initializing  LLM...\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        cache_dir="/home/iplab/LLM/models/",
        attn_implementation="eager").to('cuda')
    print("\nLLM successfully initialized.\n")

    # Configure prompt templates for different datasets
    os.makedirs(f'./results/{args.dataset_name}_processed/', exist_ok=True)
    base_prompts = {
        'truthfulqa': "Answer the question concisely. Q: {question} A:",
        'triviaqa': "Answer the question concisely. Q: {question} A:",
        'tydiqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A:",
        'coqa': "Answer the question concisely based on the context: \n {story} \n Q: {question} A:",
        'haluevaldia': "You are an assistant that answers questions concisely and accurately. Use the knowledge and conversation to respond naturally to the most recent message.\nKnowledge: {knowledge}.\nConversations: {dialogue_history}.\nYour Response:",
        'haluevalqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A:",
        'haluevalsum': "{document} \n Please summarize the above article concisely. A:",
        'combined': "Answer the question concisely. Q: {prompt} A:"
    }

    base_prompt = base_prompts.get(args.dataset_name, "")

    def process_row(row):
        if args.dataset_name in ['truthfulqa', 'triviaqa']:
            prompt = tokenizer(base_prompt.format(question=row['question']), return_tensors='pt').to("cuda") 
        elif args.dataset_name == 'tydiqa':
            prompt = tokenizer(base_prompt.format(context=row['context'], question=row['question']), return_tensors='pt').to("cuda") 
        elif args.dataset_name == 'coqa':
            prompt = tokenizer(base_prompt.format(story=row['story'], question=row['question']), return_tensors='pt').to("cuda") 
        elif args.dataset_name == 'haluevaldia':
            prompt = tokenizer(base_prompt.format(knowledge=row['knowledge'], dialogue_history=row['dialogue_history']), return_tensors='pt').to("cuda") 
        elif args.dataset_name == 'haluevalqa':
            prompt = tokenizer(base_prompt.format(context=row['knowledge'], question=row['question']), return_tensors='pt').to("cuda") 
        elif args.dataset_name == 'haluevalsum':
            prompt = tokenizer(base_prompt.format(document=row['document']),padding=True, return_tensors='pt').to("cuda")
        elif args.dataset_name == 'combined':
            prompt = tokenizer(base_prompt.format(prompt=row['prompt']), return_tensors='pt').to("cuda")

        generated = model.generate(**prompt,
                                    do_sample=False,
                                    max_new_tokens=64,
                                    pad_token_id=tokenizer.eos_token_id,
                                    return_dict_in_generate=True,
                                    output_hidden_states=True,     
                                    output_attentions=True,      
                                    output_logits=True)

        decoded = tokenizer.decode(generated.sequences[0, prompt["input_ids"].shape[-1]:],
                                    skip_special_tokens=True)
        return (
            functions.plot_internal_state_2(generated)
            + functions.plot_internal_state_2(generated, state="attention")
            + functions.probability_function(generated)
            + [decoded]
        )
    
    if args.num_workers == 1:
        result = []
        for index, row in tqdm(enumerate(dataset), total=len(dataset), desc=f"Generating responses for {args.dataset_name} dataset ..."):
            if args.dataset_name in ['truthfulqa', 'triviaqa']:
                prompt = tokenizer(base_prompt.format(question=row['question']), return_tensors='pt').to("cuda") 
            elif args.dataset_name == 'tydiqa':
                prompt = tokenizer(base_prompt.format(context=row['context'], question=row['question']), return_tensors='pt').to("cuda") 
            elif args.dataset_name == 'coqa':
                prompt = tokenizer(base_prompt.format(story=row['story'], question=row['question']), return_tensors='pt').to("cuda") 
            elif args.dataset_name == 'haluevaldia':
                prompt = tokenizer(base_prompt.format(knowledge=row['knowledge'], dialogue_history=row['dialogue_history']), return_tensors='pt').to("cuda") 
            elif args.dataset_name == 'haluevalqa':
                prompt = tokenizer(base_prompt.format(context=row['knowledge'], question=row['question']), return_tensors='pt').to("cuda") 
            elif args.dataset_name == 'haluevalsum':
                prompt = tokenizer(base_prompt.format(document=row['document']), return_tensors='pt').to("cuda") 
            elif args.dataset_name == 'combined':
                prompt = tokenizer(base_prompt.format(prompt=row['prompt']), return_tensors='pt').to("cuda")

            generated = model.generate(**prompt,
                                        do_sample=False,
                                        max_new_tokens=64,
                                        pad_token_id=tokenizer.eos_token_id,
                                        return_dict_in_generate=True,
                                        output_hidden_states=True,     
                                        output_attentions=True,      
                                        output_logits=True)

            decoded = tokenizer.decode(generated.sequences[0, prompt["input_ids"].shape[-1]:],
                                        skip_special_tokens=True)
            result.append(
                functions.plot_internal_state_2(generated)
                + functions.plot_internal_state_2(generated, state="attention")
                + functions.probability_function(generated)
                + [decoded]
            )
        
    else : 
        result = process_with_threads(args, dataset, process_row, max_workers=args.num_workers)

    # Save the results to a DataFrame
    df = pd.DataFrame(result)
    df.to_csv(f'./results/{args.dataset_name}_processed/safety_det_{args.model_name}_{args.dataset_name}_intermediate.csv', index=False)
    answers = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    print("\nResponses successfully generated.\n")
    print("=========================================================================\n")
    
    time.sleep(5)
    
    if args.dataset_name == 'combined':
        # For combined safety dataset, we already have labels
        print("Processing combined safety dataset with existing labels...\n")
        
        # Create a DataFrame with the safety labels directly from the dataset
        df_bleurt = pd.DataFrame({
            'id': range(len(dataset)),
            'safety': dataset['label']
        })
        
        # Save LLM responses with safety labels 
        pd.DataFrame({
            'prompt': dataset['prompt'][:],
            'llm_answer': answers.values,
            'type': dataset['type'][:],
            'safety': dataset['label'][:]
        }).to_csv(f'./results/{args.dataset_name}_processed/safety_det_{args.model_name}_{args.dataset_name}_responses.csv')
        
        # Prepare data for classifier training
        data = functions.data_preparation(df, df_bleurt, mode="safety")
        data.to_parquet(f'./results/{args.dataset_name}_processed/safety_det_{args.model_name}_{args.dataset_name}_dataset.pq')
        
        print("=========================================================================\n")
        
        time.sleep(5)
        
        print("Starting classifier training with the processed dataset...\n")
        trained_model = classifier.train_combined_model(data, test_size=0.25)
        torch.save(trained_model.state_dict(), f"./results/{args.dataset_name}_processed/safety_det_{args.model_name}_{args.dataset_name}_model.pth")
        
        print("\nHalluShift execution completed successfully.\n")
        print("All results and trained model have been saved in the 'result' folder.\n")    
        print("=========================================================================\n")
    else:
        print("Starting the BLEURT setup for evaluation...\n")
        # correct answers for questions 
        answer_mapping = {
            'truthfulqa': ['best_answer','question'],
            'triviaqa': ['answer','question'],
            'coqa': ['answer','question'],
            'tydiqa': ['answers','question'],
            'haluevaldia': ['right_response','dialogue_history'],
            'haluevalqa': ['right_answer','question'],
            'haluevalsum': ['right_summary','document']
        }

        if args.dataset_name in answer_mapping:
            keys = answer_mapping[args.dataset_name][:-1]
            result_dataset = pd.DataFrame([{key: d[key] for key in keys} for d in dataset])
            if args.dataset_name == 'truthfulqa':
                result_dataset['all_answers'] = result_dataset['best_answer'].apply(lambda row: [row])
            elif args.dataset_name == 'triviaqa':
                result_dataset['all_answers'] = result_dataset['answer'].apply(lambda row: row['aliases'])
            elif args.dataset_name == 'tydiqa':
                result_dataset['all_answers'] = result_dataset['answers'].apply(lambda row: row['text'])
            elif args.dataset_name == 'coqa':
                result_dataset['all_answers'] = result_dataset['answer'].apply(lambda row: [row['text']])
            else:
                result_dataset['all_answers'] = result_dataset[keys[0]].apply(lambda row: [row])
        result_dataset = pd.DataFrame({
            'answers': result_dataset['all_answers'],
            'llm_answer': answers.values,
            'id': [str(i) for i in range(len(result_dataset))]
        })
        result_dataset = result_dataset.explode('answers', ignore_index=True)

        # Installing BLEURT model
        print("Downloading BLEURT model...\n")
        if not os.path.exists("./models/BLEURT-20-D12"):
            os.system("wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip -O ./models/BLEURT-20-D12.zip")
            os.system("unzip -o ./models/BLEURT-20-D12.zip -d ./models")
        functions.column_to_txt(result_dataset, 'answers', 'answers')
        functions.column_to_txt(result_dataset, 'id', 'id')
        functions.column_to_txt(result_dataset, 'llm_answer', 'llm_answer')
        print("BLEURT model successfully downloaded\n")

        print("Running BLEURT scoring for response evaluation...\n")
        os.system(
            "python -m bleurt.score_files "
            "-candidate_file=llm_answer "
            "-reference_file=answers "
            "-bleurt_batch_size=100 "
            "-batch_same_length=True "
            "-bleurt_checkpoint=models/BLEURT-20-D12 "
            "-scores_file=scores"
        )
        print("=========================================================================\n")
        
        time.sleep(5)
        
    
        # Preparing data for training the classifier
        print("Starting Data Processing...\n")
        df_bleurt = functions.bleurt_processing("id", "scores", 0.5)
        
        # Save LLM responses with bleurt 
        pd.DataFrame({
            'questions': dataset[answer_mapping[args.dataset_name][-1]][:],
            'llm_answer': answers.values,
            'bleurt_score': df_bleurt['bleurt_score'], 
            'hallucination' : df_bleurt['hallucination']
        }).to_csv(f'./results/{args.dataset_name}_processed/hal_det_{args.model_name}_{args.dataset_name}_responses_with_bleurt.csv')
        
        data = functions.data_preparation(df, df_bleurt)
        data.to_parquet(f'./results/{args.dataset_name}_processed/hal_det_{args.model_name}_{args.dataset_name}_dataset.pq')
        
        # Remove unnecessary files
        if os.path.exists("llm_answer") and os.path.exists("answers") and os.path.exists("scores") and os.path.exists("id") :
            os.remove("llm_answer")
            os.remove("answers")
            os.remove("id")
            os.remove("scores")
        else:
            raise ValueError("BLEURT Score files not found")
        print("=========================================================================\n")
        
        time.sleep(5)
        
        print("Starting classifier training with the processed dataset...\n")
        if args.dataset_name in ['truthfulqa', 'triviaqa', 'tydiqa', 'coqa']:
            trained_model = classifier.train_combined_model(data, test_size=0.25)
        elif args.dataset_name in ['haluevaldia', 'haluevalqa', 'haluevalsum']:
            trained_model = classifier.train_combined_model(data, test_size=0.9)
        torch.save(trained_model.state_dict(), f"./results/{args.dataset_name}_processed/hal_det_{args.model_name}_{args.dataset_name}_model.pth")
        
        print("\nHalluShift execution completed successfully.\n")
        print("All results and trained model have been saved in the 'result' folder.\n")    
        print("=========================================================================\n")

if __name__ == '__main__':
    main()
