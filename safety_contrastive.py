import torch, os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import ASREvaluator, Generator, Discriminator, Intervener_safety_cont
import pyvene as pv

hf_token = "<INPUT_YOUR_HF_ACCESS_TOKEN>"

# Set device to GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Predefined HuggingFace model identifiers
HF_NAMES = {
    'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama3_8B': "meta-llama/Llama-3.1-8B",
    'Qwen2.5_7B' : "Qwen/Qwen2.5-7B", 
    'Yi1.5_9B' : "01-ai/Yi-1.5-9B"
}

# Global results dataframe to accumulate evaluation outputs
df_results = []
        
def formatter(dataset_name, dataset, tokenizer): 
    """
    Formats and tokenizes prompts from the dataset.
    """
    base_prompts = {
        'jailbreak-bench' : "{prompt}.",
        'advbench' : "{prompt}.",
    }
    base_prompt = base_prompts.get(dataset_name, "{prompt}")
    all_prompts = []  
    for _, row in enumerate(dataset):
        if dataset_name in ['malicious-instruct','advbench','trustllm']:
            prompt = tokenizer(base_prompt.format(prompt=row['prompt']), truncation=True, max_length=128, return_tensors='pt')
        elif dataset_name == 'jailbreak-bench':
            prompt = tokenizer(base_prompt.format(prompt=row['Goal']), truncation=True, max_length=128, return_tensors='pt')   
        all_prompts.append(prompt)
    return all_prompts

def get_answers(i, test_idxs, dataset_name, questions, dataset,  model=None,
                tokenizer=None, verbose=False, device=DEVICE):
    """
    Generates model responses for prompts.
    """
    frame = dataset.to_pandas()
    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(questions, desc=f"Getting answers for fold {i}")):
            if idx not in test_idxs:
                continue
            inputs = inputs.to(device)
            _, output = model.generate(inputs, top_k=1, max_new_tokens=64, num_return_sequences=1,)
            model_gen_tokens = output[:, inputs.input_ids.shape[-1]:]
            model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
            model_gen_str = model_gen_str.strip()
            if verbose: 
                tqdm.write(f"MODEL_OUTPUT: {model_gen_str}")
            frame.loc[idx, 'intervened_answers'] = model_gen_str        
    if device:
        torch.cuda.empty_cache()
    if dataset_name in ['malicious-instruct','advbench','trustllm']:
        frame = frame[['prompt', 'intervened_answers']]
    elif dataset_name == 'jailbreak-bench':
        frame = frame[['Goal', 'intervened_answers']].rename(columns={'Goal': 'prompt'})
    tqdm.write('\n___________________________________________________________________\n')
    return frame.dropna(subset=['intervened_answers'])

def evaluation(i, questions, dataset, tokenizer, test_idxs, dataset_name, models, device=DEVICE, verbose=False): 
    """
    Evaluates a model on a given dataset split.
    """
    for mdl in models.keys(): 
        llama_model = models[mdl]
        questions = get_answers(i, test_idxs, dataset_name, questions, dataset, model=llama_model, tokenizer=tokenizer, device=device, verbose=verbose)
        df_results.append(questions)

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
    tqdm.write(f"Training probes for fold {i}")
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
    """Select top-k layers with highest probe accuracy."""
    probes, all_head_accs_np = train_probes(i, seed, train_idxs, val_idxs, activations, labels,
                                            num_layers=num_layers)
    top_accs = np.argsort(all_head_accs_np.reshape(num_layers))[::-1][:num_to_intervene]
    return top_accs, probes

def wrapper(intervener):
    """
    Wraps an intervention function for PyVene compatibility.
    """
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

def load_trust_dataset():
    """
    Downloads and processes the TrustLLM dataset.
    Returns:
        Dataset: The processed TrustLLM dataset.
    """
    from huggingface_hub import hf_hub_download
    import shutil
    save_path = './llm_trust_dataset'
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(f"llm_trust_dataset/misuse.json"):
        try:
            file_path = hf_hub_download(repo_id="TrustLLM/TrustLLM-dataset", filename="safety/misuse.json", repo_type="dataset")
            shutil.copy(file_path, os.path.join(os.getcwd(), "llm_trust_dataset/misuse.json"))
        except Exception as e:
            print(f"Failed to download coqa dataset file: {e}")
    
    with open("llm_trust_dataset/misuse.json", "r") as file:
        data = json.load(file)  
        return Dataset.from_list(data)
    
def contrastive_loss(anchor, positive, negative, margin=1.0):
    """
    Computes a contrastive loss.

    """    
    anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
    positive = torch.nn.functional.normalize(positive, p=2, dim=1)
    negative = torch.nn.functional.normalize(negative, p=2, dim=1)
    pos_sim = torch.nn.functional.cosine_similarity(anchor, positive)
    neg_sim = torch.nn.functional.cosine_similarity(anchor, negative)
    # Contrastive loss (margin ranking loss)
    loss = torch.clamp(margin - pos_sim + neg_sim, min=0.0)
    return loss.mean()

def train_contrastive_adversarial(i, HIDDEN_DIM, dataloader, num_epochs=100, lr = 1e-5, lambda_recon=1e-5, lambda_contrastive =  0.5, device=DEVICE):
    """
    Trains a Generator-Discriminator pair using contrastive adversarial loss.
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
    for _ in tqdm(range(num_epochs), desc=f"Training adversarial network for fold {i}"):
        for base_batch, aligned_batch, positive_batch, negative_batch in dataloader:
            base_batch = base_batch.to(device)
            aligned_batch = aligned_batch.to(device)
            positive_batch = positive_batch.to(device)
            negative_batch = negative_batch.to(device)
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
            g_contrastive_loss = contrastive_loss(fake_batch, positive_batch, negative_batch)
            # torch.clamp(mse(fake_batch, aligned_batch), max=1.0)
            g_loss = g_adv_loss + lambda_recon * g_recon_loss + lambda_contrastive * g_contrastive_loss
            g_opt.zero_grad()
            g_loss.backward()
            # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            g_opt.step()

        # Step learning rate scheduler
        g_sched.step()
        d_sched.step()
        # current_g_lr, current_d_lr = g_opt.param_groups[0]['lr'], d_opt.param_groups[0]['lr']
        # tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] D_loss: {d_loss.item():.6f} | G_loss: {g_loss.item():.6f} | GLR: {current_g_lr:.6f} | DLR: {current_d_lr:.6f}")
    tqdm.write('')
    return generator, discriminator
    
    
def main(): 
    """
    Main function to run ARREST-Contrastive training and evaluation pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='malicious-instruct', help='Dataset')
    parser.add_argument('--num_layers', type=int, default=1, help='number of top layers to intervene on') 
    parser.add_argument('--alpha', type=float, default=1, help='intervention strength') 
    parser.add_argument("--num_fold", type=int, default=5, help="number of folds") 
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate') 
    parser.add_argument('--lambda_contrastive', type=float, default=0.5, help='Contrastive loss')
    parser.add_argument('--lambda_recon', type=float, default=0, help='Reconstruction loss') 
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    print(f"""\n
    ==================================================================================
                  ARREST-Contrastive for Safety Execution Started
    ==================================================================================
    Dataset: {args.dataset_name}    Model: {args.model_name}    Folds: {args.num_fold}

    Workflow:
    1. Training probes on each layer of LLM to find maximum concept misalignment layer
    2. Contrastive training on the chosen layer
    3. Using the Generator during inference to improve safety

    Output: 
    - ASR(%) percentage of unsafe answers displayed on the screen
    - LLM responses are stored in 'safety/responses' folder
    ==================================================================================\n
    """)
    
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load and prepare dataset
    print("\nDownloading Dataset...\n")
    if args.dataset_name == 'malicious-instruct':
        dataset = load_dataset("walledai/MaliciousInstruct")['train']
    elif args.dataset_name == 'advbench':
        dataset  =  load_dataset("walledai/AdvBench")['train']
    elif args.dataset_name == 'jailbreak-bench':
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors") 
        dataset  =  ds['harmful']
    elif args.dataset_name == 'trustllm':
        dataset = load_trust_dataset()
    else: 
        raise ValueError("Invalid dataset name")
    
    fold_idxs = np.array_split(np.arange(len(dataset)), args.num_fold)

    # Load model and tokenizer
    print("Initializing  LLM...\n")
    MODEL = HF_NAMES[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=hf_token)
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

    # Load hidden activations and labels
    activations_base = np.load(f"./safety/hidden/{args.model_name}_{args.dataset_name}_head_wise.npy")
    activations_aligned = np.load(f"./safety/hidden/{args.model_name}_aligned_{args.dataset_name}_head_wise.npy")
    
    labels_base = np.ones(activations_base.shape[0])
    labels_aligned = np.zeros(activations_aligned.shape[0])
    
    activations_positive = np.load(f"./safety/hidden/{args.model_name}_{args.dataset_name}_head_wise_positive.npy")
    activations_negative = np.load(f"./safety/hidden/{args.model_name}_{args.dataset_name}_head_wise_negative.npy")
    
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
        activations_training = np.concatenate([activations_base[train_idxs], activations_aligned[train_idxs]], axis=0)
        labels = np.concatenate([labels_base[train_idxs], labels_aligned[train_idxs]], axis=0)
        perm = np.random.permutation(len(labels))
        activations_training = activations_training[perm]
        labels = labels[perm]

        train_set_idxs = np.random.choice(np.arange(len(train_idxs)), size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in np.arange(len(train_idxs)) if x not in train_set_idxs])
        
        # Training probes on each LLM layers
        top_layers, _ = get_top_layers(i+1, train_set_idxs, val_set_idxs, activations_training, labels, num_layers,
                                        args.seed, args.num_layers)

        # Create pyvene config
        interveners = []
        pv_config = []
        for layer in top_layers :
            aligned_activations = activations_aligned[train_idxs,layer,:]
            base_activations = activations_base[train_idxs,layer,:]
            positive_activations = activations_positive[train_idxs,layer,:]
            negative_activations = activations_negative[train_idxs,layer,:]
            train_dataset = TensorDataset(torch.tensor(base_activations), torch.tensor(aligned_activations), torch.tensor(positive_activations), torch.tensor(negative_activations))
            dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            generator, discriminator = train_contrastive_adversarial(i+1, hidden_size, dataloader, num_epochs=args.num_epochs, lr=args.lr, lambda_recon=args.lambda_recon, lambda_contrastive=args.lambda_contrastive, device=DEVICE)
            intervener = Intervener_safety_cont(generator, discriminator, args.alpha)
            interveners.append(intervener)
            pv_config.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(intervener),
            })
        intervened_model = pv.IntervenableModel(pv_config, model)

        filename = f'{args.dataset_name}_{args.model_name}_top{args.num_layers}layers_alpha{args.alpha}_advcontr'
        
        tqdm.write('')
        evaluation(
            i+1,
            prompts,
            dataset,
            tokenizer,
            test_idxs,
            args.dataset_name,
            models={args.model_name: intervened_model},
            device=DEVICE
        )
    return filename


if __name__ == "__main__":
    filename = main()
    df_results = pd.concat(df_results, ignore_index=True)
    evaluator = ASREvaluator()
    evaluator._process_file(filename, df_results)