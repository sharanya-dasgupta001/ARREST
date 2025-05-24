import functions
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

# Suppress warnings
warnings.filterwarnings("ignore")

# Mapping of model names to their identifiers
MODELS_NAMES = {
    'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama3_8B': "meta-llama/Llama-3.1-8B",
    'opt6.7B': "facebook/opt-6.7b",
    'llama2_7B_chat' : "meta-llama/Llama-2-7b-chat-hf"
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

def load_dataset_by_name(file):
    """Loads a dataset based on the provided name.
    Args:
        dataset_name (str): The name of the dataset to load.
    Returns:
        Dataset: The loaded dataset.
    """
    
    # Load the TruthfulQA dataset's validation split
    if "truthfulqa" in file.name.lower():
        return "truthfulqa", load_dataset("truthful_qa", 'generation')['validation']
    
    # Load the TriviaQA dataset and remove duplicate questions
    elif 'triviaqa' in file.name.lower():
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()
        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch
        return "triviaqa", dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    
    # Load the TyDiQA dataset and filter for English questions
    elif 'tydiqa' in file.name.lower():
        dataset = load_dataset("tydiqa", "secondary_task", split="train")
        return "tydiqa", dataset.filter(lambda row: "english" in row["id"])
    
    # Load the CoQA dataset
    elif 'coqa' in file.name.lower():
        return "coqa", load_coqa_dataset()
    
    # Load a specific subset of the HaluEval dataset
    elif 'haluevaldia' in file.name.lower():
        return "haluevaldia", load_dataset("pminervini/HaluEval", "dialogue")['data']
    elif 'haluevalqa' in file.name.lower():
        return "haluevalqa", load_dataset("pminervini/HaluEval", "qa")['data']
    elif 'haluevalsum' in file.name.lower():
        return "haluevalsum", load_dataset("pminervini/HaluEval", "summarization")['data']
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

def bleurt(file):
    """
    Main function to perform the following tasks:
    - Parse command-line arguments.
    - Download and preprocess datasets.
    - Load and configure language models.
    - Generate responses using the language model.
    - Evaluate generated responses using BLEURT.
    - Train classifier.
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default='llama2_7B_chat', help='Name of the model to use.')
    # parser.add_argument('--dataset_name', type=str, default='truthfulqa', help='Name of the dataset to use.')
    # parser.add_argument('--num_workers', type=int, default=4, help='Number of cpu threads to use.')
    # # parser.add_argument('--filename', type=str, default=4, help='Number of cpu threads to use.')
    # args = parser.parse_args()

    # Set random seed for reproducibility
    seed_everything(42)
    
    dataset_name, dataset = load_dataset_by_name(file)
    df = pd.read_csv(file)
    
    answers = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    
    
    print("Starting the BLEURT setup for evaluation...\n")
    # correct answers for questions 
    answer_mapping = {
        'truthfulqa': ['best_answer', 'correct_answers','question'],
        'triviaqa': ['answer','question'],
        'coqa': ['answer','question'],
        'tydiqa': ['answers','question'],
        'haluevaldia': ['right_response','dialogue_history'],
        'haluevalqa': ['right_answer','question'],
        'haluevalsum': ['right_summary','document']
    }

    if dataset_name in answer_mapping:
        keys = answer_mapping[dataset_name][:-1]
        result_dataset = pd.DataFrame([{key: d[key] for key in keys} for d in dataset])
        if dataset_name == 'truthfulqa':
            result_dataset['all_answers'] = result_dataset.apply(lambda row: [row['best_answer']] + row['correct_answers'],axis=1)
        elif dataset_name == 'triviaqa':
            result_dataset['all_answers'] = result_dataset['answer'].apply(lambda row: row['aliases'])
        elif dataset_name == 'tydiqa':
            result_dataset['all_answers'] = result_dataset['answers'].apply(lambda row: row['text'])
        elif dataset_name == 'coqa':
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
    if not os.path.exists("/home/iplab/LLM/models/BLEURT-20-D12"):
        os.system("wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip -O /home/iplab/LLM/models/BLEURT-20-D12.zip")
        os.system("unzip -o /home/iplab/LLM/models/BLEURT-20-D12.zip -d /home/iplab/LLM/models/")
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
        "-bleurt_checkpoint=/home/iplab/LLM/models/BLEURT-20-D12 "
        "-scores_file=scores"
    )
    print("=========================================================================\n")
    
    time.sleep(5)
    
    # Preparing data for training the classifier
    print("Starting Data Processing...\n")
    df_bleurt = functions.bleurt_processing("id", "scores", 0.5)
    
    # Save LLM responses with bleurt 
    df_bleurt = pd.DataFrame({
        'questions': dataset[answer_mapping[dataset_name][-1]][:],
        'llm_answer': answers.values,
        'bleurt_score': df_bleurt['bleurt_score'], 
        'hallucination' : df_bleurt['hallucination']
    })
    df_bleurt.to_csv(f'/home/iplab/LLM/mitigation_results/responses_hal/{file.name}_bleurt.csv')
    print(df_bleurt['hallucination'].value_counts())

    
    # Remove unnecessary files
    if os.path.exists("llm_answer") and os.path.exists("answers") and os.path.exists("scores") and os.path.exists("id") :
        os.remove("llm_answer")
        os.remove("answers")
        os.remove("id")
        os.remove("scores")
    else:
        raise ValueError("BLEURT Score files not found")
    print("=========================================================================\n")
    
from pathlib import Path
def processing(folderpath):
    directory = Path(folderpath)
    for file in directory.iterdir():
        if file.is_file() and file.suffix == '.csv':
            bleurt(file)

if __name__ == '__main__':
    import sys

# Run as: python script.py Alice
    folder_name = sys.argv[1]
    processing(folder_name)