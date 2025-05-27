import torch, os, json
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pyvene as pv
from utils import BleurtEvaluator, Generator, Discriminator, Intervener_hallucination

hf_token = "<INPUT_YOUR_HF_ACCESS_TOKEN>"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping of model aliases to Hugging Face model names
HF_NAMES = {
    'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama3_8B': "meta-llama/Llama-3.1-8B",
    'vicuna_7B' : "lmsys/vicuna-7b-v1.5"
}

# Global results dataframe to accumulate evaluation outputs
df_results = [] 

        
def formatter(dataset_name, dataset, tokenizer): 
    """
    Format input prompts from the dataset using dataset-specific templates.
    """
    base_prompts = {
        'truthfulqa': "Answer the question concisely. Q: {question} A:",
        'triviaqa': "Answer the question concisely. Q: {question} A:",
        'tydiqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A:",
        'coqa': "Answer the question concisely based on the context: \n {story} \n Q: {question} A:"
    }

    base_prompt = base_prompts.get(dataset_name, "{prompt}")
    all_prompts = []

        
    for _, row in enumerate(dataset):
        if dataset_name in ['truthfulqa', 'triviaqa']:
            prompt = tokenizer(base_prompt.format(question=row['question']), truncation=True, max_length=128, return_tensors='pt')
        elif dataset_name == 'tydiqa':
            prompt = tokenizer(base_prompt.format(context=row['context'], question=row['question']), truncation=True, max_length=128, return_tensors='pt')
        elif dataset_name == 'coqa':
            prompt = tokenizer(base_prompt.format(story=row['story'], question=row['question']),truncation=True, max_length=128,  return_tensors='pt')
        all_prompts.append(prompt)
        
    return all_prompts

def get_answers(i, test_idxs, dataset_name, questions, dataset,  model=None,
                tokenizer=None, verbose=False, device=DEVICE):
    """
    Generate model responses to questions in the dataset.
    """
    frame = dataset.to_pandas()
    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(questions, desc=f"Getting answers for fold {i+1}")):
            if idx not in test_idxs:
                    continue
            try :
                inputs = inputs.to(device)
                _, output = model.generate(inputs, top_k=1, max_new_tokens=64, num_return_sequences=1,)
                model_gen_tokens = output[:, inputs.input_ids.shape[-1]:]
                model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
                model_gen_str = model_gen_str.strip()
                if verbose: 
                    tqdm.write(f"MODEL_OUTPUT: {model_gen_str}")
                
                frame.loc[idx, 'intervened_answers'] = model_gen_str
            except :
                torch.cuda.empty_cache()
                print(f"\nOut of memory error, skipping {idx}\n")
                frame.loc[idx, 'intervened_answers'] = ""
                continue
                
    if device:
        torch.cuda.empty_cache()
        
    if dataset_name in ['truthfulqa', 'triviaqa', 'coqa','tydiqa']:
        frame = frame[['question', 'intervened_answers']].rename(columns={'question': 'prompt'})
    tqdm.write('\n___________________________________________________________________\n')
    return frame.dropna(subset=['intervened_answers'])

def evaluation(i, questions, dataset, tokenizer, test_idxs, dataset_name, models, device=DEVICE, verbose=False): 
    """
    Evaluate models on the dataset by generating answers and storing results.
    """
    for mdl in models.keys(): 
        llama_model = models[mdl]
        answers = get_answers(i, test_idxs, dataset_name, questions, dataset, model=llama_model, tokenizer=tokenizer, device=device, verbose=verbose)
        df_results.append(answers)

def train_probes(i, seed, train_set_idxs, val_set_idxs, activations_training, labels, num_layers):
    """
    Train probes on each layer to distinguish aligned vs non aligned hidden states.
    """
    all_head_accs = []
    probes = []
    all_X_train = activations_training[train_set_idxs]
    all_X_val = activations_training[val_set_idxs]
    y_train = labels[train_set_idxs]
    y_val = labels[val_set_idxs]
    tqdm.write(f"Training probes for fold {i+1}")
    for layer in range(num_layers): 
        X_train = all_X_train[:,layer,:]
        X_val = all_X_val[:,layer,:]
        clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)
        all_head_accs.append(accuracy_score(y_val, y_val_pred))
        probes.append(clf)
    all_head_accs_np = np.array(all_head_accs)
    tqdm.write('')
    return probes, all_head_accs_np

def get_top_layers(i, train_idxs, val_idxs, activations, labels, num_layers,
                seed, num_to_intervene):
    """
    Select top-k layers with highest probe accuracy.
    """
    probes, all_head_accs_np = train_probes(i, seed, train_idxs, val_idxs, activations, labels,
                                            num_layers=num_layers)
    top_accs = np.argsort(all_head_accs_np.reshape(num_layers))[::-1][:num_to_intervene]
    return top_accs, probes

def wrapper(intervener):
    """Wraps an intervention function."""
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped


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
            print(f"\nFailed to download coqa dataset file: {e}\n")
    
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
    
    
def train_adversarial(i, HIDDEN_DIM, dataloader, num_epochs=100, lr = 1e-5, lambda_recon=1e-5, device=DEVICE):
    """
    Trains generator and discriminator adversarially on hidden activations.
    """ 
    generator = Generator(HIDDEN_DIM).to(device)
    discriminator = Discriminator(HIDDEN_DIM).to(device)
    
    g_opt = optim.AdamW(generator.parameters(), lr=lr)
    d_opt = optim.AdamW(discriminator.parameters(), lr=lr)

    g_sched = optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=num_epochs, eta_min=1e-5)
    d_sched = optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=num_epochs, eta_min=1e-5)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    generator.train()
    discriminator.train()

    for _ in tqdm(range(num_epochs), desc =f"Training adversarial network for fold {i+1}"):
        for base_batch, aligned_batch in dataloader:
            base_batch = base_batch.to(device)
            aligned_batch = aligned_batch.to(device)
            
            with torch.no_grad():
                fake_batch = generator(base_batch)

            real_labels = torch.ones(aligned_batch.size(0), 1).to(device)
            fake_labels = torch.zeros(base_batch.size(0), 1).to(device)

            d_real = discriminator(aligned_batch)
            d_fake = discriminator(fake_batch)

            d_loss_real = bce(d_real, real_labels)
            d_loss_fake = bce(d_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()
            
            fake_batch = generator(base_batch)
            d_fake = discriminator(fake_batch)

            g_adv_loss = bce(d_fake, real_labels)
            g_recon_loss = mse(fake_batch, aligned_batch)
            # torch.clamp(mse(fake_batch, aligned_batch), max=1.0)
            g_loss = g_adv_loss + lambda_recon * g_recon_loss

            g_opt.zero_grad()
            g_loss.backward()
            # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            g_opt.step()

        # Step learning rate scheduler
        # g_sched.step()
        # d_sched.step()
        # current_g_lr, current_d_lr = g_opt.param_groups[0]['lr'], d_opt.param_groups[0]['lr']
        # tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] D_loss: {d_loss.item():.6f} | G_loss: {g_loss.item():.6f} | GLR: {current_g_lr:.6f} | DLR: {current_d_lr:.6f}")
    tqdm.write('')
    return generator, discriminator
    
    
def main(): 
    """
    Main pipeline for training and evaluating ARREST-Adversarial for hallucination.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='truthfulqa', help='Dataset', choices=['truthfulqa', 'triviaqa', 'tydiqa', 'coqa'])
    parser.add_argument('--num_layers', type=int, default=1, help='number of top layers to intervene on') 
    parser.add_argument('--alpha', type=float, default=1.0, help='intervention strength') 
    parser.add_argument("--num_fold", type=int, default=5, help="number of folds")
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lambda_recon', type=float, default=1e-5, help='Reconstruction loss')
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    print(f"""\n
    ==================================================================================
                  ARREST-Adversarial for Hallucination Execution Started
    ==================================================================================
    Dataset: {args.dataset_name}    Model: {args.model_name}    Folds: {args.num_fold}

    Workflow:
    1. Training probes on each layer of LLM to find maximum concept misalignment layer
    2. Adversarial training on the chosen layer
    3. Using the Generator during inference to improve truthfulness

    Output: 
    - Truth(%) percentage of truthful answers displayed on the screen
    - LLM responses are stored in 'hallucination/responses' folder
    ==================================================================================\n
    """)
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Load dataset
    print("\nDownloading Dataset...\n")
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
        dataset = dataset.map(lambda example: {'answer': example['answer']['text']})
    else: 
        raise ValueError("Invalid dataset name")
    
    fold_idxs = np.array_split(np.arange(len(dataset)), args.num_fold)

    # Load model
    print("Initializing  LLM...\n")
    MODEL = HF_NAMES[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=hf_token)
    os.makedirs('./models', exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        use_auth_token=hf_token,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        cache_dir="./models",
        attn_implementation="eager").to(DEVICE)
    
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # Tokenize prompts
    print("\nTokenizing prompts...\n")
    prompts = formatter(args.dataset_name, dataset, tokenizer)         
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Load hidden states
    activations_hal = np.load(f"./hallucination/hidden/{args.model_name}_{args.dataset_name}_head_wise.npy")
    activations_truth = np.load(f"./hallucination/hidden/{args.model_name}_{args.dataset_name}_head_wise_truth.npy")
    
    labels_hal = np.zeros(activations_hal.shape[0])
    labels_truth = np.ones(activations_truth.shape[0])
    
    # Run different folds
    for i in range(args.num_fold):
        tqdm.write(f"Running folds {i+1}\n")
        if args.num_fold == 1 :
            train_idxs = fold_idxs[0]
            test_idxs = fold_idxs[0]
        else :
            train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
            test_idxs = fold_idxs[i]
        
        # Prepare train/val split
        activations_training = np.concatenate([activations_hal[train_idxs], activations_truth[train_idxs]], axis=0)
        labels = np.concatenate([labels_hal[train_idxs], labels_truth[train_idxs]], axis=0)
        perm = np.random.permutation(len(labels))
        activations_training = activations_training[perm]
        labels = labels[perm]

        train_set_idxs = np.random.choice(np.arange(len(train_idxs)), size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in np.arange(len(train_idxs)) if x not in train_set_idxs])
        
        # Training probes on each LLM layers
        top_layers, _ = get_top_layers(i, train_set_idxs, val_set_idxs, activations_training, labels, num_layers, args.seed, args.num_layers)

        # Create pyvene config
        pv_config = []
        for layer in top_layers :
            hal_activations = activations_hal[train_idxs,layer,:]
            truth_activations = activations_truth[train_idxs,layer,:]
            train_dataset = TensorDataset(torch.tensor(hal_activations), torch.tensor(truth_activations))
            dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            generator, discriminator = train_adversarial(i, hidden_size, dataloader, num_epochs=args.num_epochs, lr=args.lr, lambda_recon=args.lambda_recon, device=DEVICE)
            intervener = Intervener_hallucination(generator, discriminator, args.alpha) 
            pv_config.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(intervener),
            })
        intervened_model = pv.IntervenableModel(pv_config, model)   
        tqdm.write('')                   
        evaluation(
            i,
            prompts,
            dataset,
            tokenizer,
            test_idxs,
            args.dataset_name,
            models={args.model_name: intervened_model},
            device=DEVICE
        )
        
    filename = f'{args.dataset_name}_{args.model_name}_top{args.num_layers}layers_alpha{args.alpha}_recon{args.lambda_recon}_adv'
    return filename


if __name__ == "__main__":
    filename = main()
    df_results = pd.concat(df_results, ignore_index=True)
    evaluator = BleurtEvaluator()
    evaluator.run_bleurt(filename, df_results)
    