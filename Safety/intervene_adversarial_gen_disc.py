import torch
import numpy as np
import pandas as pd
import os, json
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pyvene as pv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_NAMES = {
   'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama3_8B': "meta-llama/Llama-3.1-8B",
    'opt6.7B': "facebook/opt-6.7b",
    'Qwen2.5_7B' : "Qwen/Qwen2.5-7B", 
    'Yi1.5_9B' : "01-ai/Yi-1.5-9B", 
    'vicuna_7B' : "lmsys/vicuna-7b-v1.5"
}
df_results = []
class Generator(nn.Module):
    def __init__(self, hidden_dim=4096):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        return self.model(x)

class DiscriminatorV1(nn.Module):
    def __init__(self, hidden_dim=4096):
        super(DiscriminatorV1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1) 
        )

    def forward(self, x):
        return self.model(x)
    
class DiscriminatorV2(nn.Module):
    def __init__(self, hidden_dim=4096):
        super(DiscriminatorV2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, 1,  bias=False)
        )
    def forward(self, x):
        return self.model(x)
        
def formatter(model_name, dataset_name, dataset, tokenizer): 
    
    base_prompts = {
        'truthfulqa': "Answer the question concisely. Q: {question} A:",
        'triviaqa': "Answer the question concisely. Q: {question} A:",
        'tydiqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A:",
        'coqa': "Answer the question concisely based on the context: \n {story} \n Q: {question} A:",
        'haluevaldia': "You are an assistant that answers questions concisely and accurately. Use the knowledge and conversation to respond naturally to the most recent message.\nKnowledge: {knowledge}.\nConversations: {dialogue_history}.\nYour Response:",
        'haluevalqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A:",
        'haluevalsum': "{document} \n Please summarize the above article concisely. A:",
        'jailbreak-bench' : "{prompt}.",
        'advbench' : "{prompt}.",
    }

    base_prompt = base_prompts.get(dataset_name, "{prompt}")
    all_prompts = []
    
    if model_name in  ['llama2_7B_chat']:
        def format_prompt(user_input):
            return f"[INST] {user_input} [/INST]"
    else :
        def format_prompt(user_input):
            return user_input
        
    for index, row in enumerate(dataset):
        if dataset_name == 'sorry-Bench' :
            prompt = tokenizer(format_prompt(base_prompt.format(prompt=row['turns'][0])), return_tensors='pt')
        elif dataset_name in ['over-refusal','malicious-instruct','advbench','trustllm']:
            prompt = tokenizer(format_prompt(base_prompt.format(prompt=row['prompt'])), return_tensors='pt')
        elif dataset_name == 'jailbreak-bench':
            prompt = tokenizer(format_prompt(base_prompt.format(prompt=row['Goal'])),return_tensors='pt')
            
        all_prompts.append(prompt)
        
    return all_prompts

def run_answers(test_idxs, model_name, dataset_name, questions, dataset,  model=None,
                tokenizer=None, verbose=False, device=DEVICE, instruction_prompt="default"):
    frame = dataset.to_pandas()
    # tokens = formatter(model_name, dataset_name, dataset, tokenizer)

    sequences = []
    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(questions, desc="run_answers")):
            if idx not in test_idxs:
                continue
            # max_len = inputs.input_ids.shape[-1] + 64
            inputs = inputs.to(device)
            _, output = model.generate(inputs, top_k=1, max_new_tokens=64, num_return_sequences=1,)
            model_gen_tokens = output[:, inputs.input_ids.shape[-1]:]
            model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
            model_gen_str = model_gen_str.strip()
            if verbose: 
                print("MODEL_OUTPUT: ", model_gen_str)
            
            frame.loc[idx, 'intervened_answers'] = model_gen_str
            sequences.append(model_gen_str)

    if device:
        torch.cuda.empty_cache()

    if dataset_name == 'sorry-Bench' :
        frame['turns'] = frame['turns'].apply(lambda x: x[0])
        frame = frame[['turns', 'intervened_answers']].rename(columns={'turns': 'prompt'})
    elif dataset_name in ['over-refusal','malicious-instruct','advbench','trustllm']:
        frame = frame[['prompt', 'intervened_answers']]
    elif dataset_name == 'jailbreak-bench':
        frame = frame[['Goal', 'intervened_answers']].rename(columns={'Goal': 'prompt'})
        
        
    return frame.dropna(subset=['intervened_answers'])

def alt_evaluate(questions, dataset, tokenizer, test_idxs, model_name, dataset_name, models, output_path, device=DEVICE,
                    verbose=False, instruction_prompt="default"): 
        
    for mdl in models.keys(): 

        llama_model = models[mdl]
        # llama_tokenizer = AutoTokenizer.from_pretrained(HF_NAMES[mdl])
        questions = run_answers(test_idxs, model_name, dataset_name, questions, dataset, model=llama_model, tokenizer=tokenizer,
                        device=device, verbose=verbose,
                        instruction_prompt=instruction_prompt)

        df_results.append(questions)
        # questions.to_csv(output_path)

def get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, head_wise_activations, labels): 

    com_directions = []

    for layer in tqdm(range(num_layers), desc="get_com_directions"): 
        for head in range(num_heads): 
            usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
            usable_head_wise_activations = head_wise_activations[:,layer,head,:][usable_idxs]
            usable_labels = labels[usable_idxs]
            true_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions

def train_probes(seed, train_set_idxs, val_set_idxs, head_wise_activations, labels, num_layers,
                num_heads):
    
    all_head_accs = []
    probes = []

    all_X_train = head_wise_activations[train_set_idxs]
    all_X_val = head_wise_activations[val_set_idxs]
    y_train = labels[train_set_idxs]
    y_val = labels[val_set_idxs]

    for layer in tqdm(range(num_layers), desc="train_probes"): 
        X_train = all_X_train[:,layer,:]
        X_val = all_X_val[:,layer,:]

        clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)
        all_head_accs.append(accuracy_score(y_val, y_val_pred))
        probes.append(clf)

    all_head_accs_np = np.array(all_head_accs)

    return probes, all_head_accs_np

def train_probes2(seed, train_set_idxs, val_set_idxs, head_wise_activations, 
                labels, num_layers, num_heads):

    all_head_accs = []
    weight_vectors = []

    all_X_train = head_wise_activations[train_set_idxs]
    all_X_val = head_wise_activations[val_set_idxs]
    y_train = labels[train_set_idxs]
    y_val = labels[val_set_idxs]

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    for layer in tqdm(range(num_layers), desc="train_probes"): 
        for head in range(num_heads): 
            X_train = all_X_train[:, layer, head, :]
            X_val = all_X_val[:, layer, head, :]

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)

            input_dim = X_train.shape[1]

            # Linear model without bias
            model = torch.nn.Linear(input_dim, 1, bias=False).to(DEVICE)
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # Train the model
            for epoch in range(100):  # fixed number of epochs
                model.train()
                optimizer.zero_grad()
                logits = model(X_train_tensor).squeeze()
                loss = criterion(logits, y_train_tensor)
                loss.backward()
                optimizer.step()

            # Evaluate accuracy on validation set
            model.eval()
            with torch.no_grad():
                logits_val = model(X_val_tensor).squeeze()
                preds_val = (torch.sigmoid(logits_val) > 0.5).cpu().numpy()
                acc = accuracy_score(y_val, preds_val)
                all_head_accs.append(acc)

                # Extract weight vector (no bias term)
                weight_vector = model.weight.detach().cpu().numpy().squeeze()
                weight_vectors.append(weight_vector)

    all_head_accs_np = np.array(all_head_accs)

    return np.array(weight_vectors), all_head_accs_np

def get_top_layers(train_idxs, val_idxs, activations, labels, num_layers, num_heads,
                seed, num_to_intervene, use_random_dir=False):
    
    probes, all_head_accs_np = train_probes(seed, train_idxs, val_idxs, activations, labels,
                                            num_layers=num_layers, num_heads=num_heads)

    top_accs = np.argsort(all_head_accs_np.reshape(num_layers))[::-1][:num_to_intervene]
    return top_accs, probes

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class ISI_Intervener():
    collect_state = True
    collect_action = True
    attr_idx = -1
    def __init__(self, generator, discriminator, multiplier):
        self.multiplier = multiplier
        self.generator = generator
        self.direction = discriminator.model[0].weight.view(-1)
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
        self.generator = None
        self.direction = None
    def __call__(self, b, s): 
        self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        self.generator.to(b.dtype).to(b.device)
        self.direction.to(b.dtype).to(b.device)
        self.generator.eval()
        with torch.no_grad():
            b[0, -1] = self.multiplier * self.generator(b[0, -1]) + (1 - self.multiplier) * self.direction
        return b

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

def train_adversarial(HIDDEN_DIM, dataloader, num_epochs=50, lr = 1e-5, lambda_recon=1e-5,
                        scheduler_type="cosine", step_size=5, gamma=0.5, device=DEVICE):
        
    generator = Generator(HIDDEN_DIM).to(device)
    discriminator = DiscriminatorV2(HIDDEN_DIM).to(device)
    
    g_opt = optim.Adam(generator.parameters(), lr=lr)
    d_opt = optim.Adam(discriminator.parameters(), lr=lr)

    if scheduler_type == "step":
        g_sched = optim.lr_scheduler.StepLR(g_opt, step_size=step_size, gamma=gamma)
        d_sched = optim.lr_scheduler.StepLR(d_opt, step_size=step_size, gamma=gamma)
    elif scheduler_type == "cosine":
        g_sched = optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=num_epochs, eta_min=1e-5)
        d_sched = optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=num_epochs, eta_min=1e-5)
    else:
        g_sched = optim.lr_scheduler.ReduceLROnPlateau(g_opt, mode='min', factor=0.5, patience=5)
        d_sched = optim.lr_scheduler.ReduceLROnPlateau(d_opt, mode='min', factor=0.5, patience=5)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    generator.train()
    discriminator.train()

    for epoch in tqdm(range(num_epochs)):
        for base_batch, chat_batch in dataloader:
            base_batch = base_batch.to(device)
            chat_batch = chat_batch.to(device)
            # ================================ Train Discriminator ================================
            with torch.no_grad():
                fake_batch = generator(base_batch)

            real_labels = torch.ones(chat_batch.size(0), 1).to(device)
            fake_labels = torch.zeros(base_batch.size(0), 1).to(device)

            d_real = discriminator(chat_batch)
            d_fake = discriminator(fake_batch)

            d_loss_real = bce(d_real, real_labels)
            d_loss_fake = bce(d_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # ================================ Train Generator ================================
            
            fake_batch = generator(base_batch)
            d_fake = discriminator(fake_batch)

            g_adv_loss = bce(d_fake, real_labels)
            g_recon_loss = mse(fake_batch, chat_batch)
            # torch.clamp(mse(fake_batch, chat_batch), max=1.0)
            g_loss = g_adv_loss + lambda_recon * g_recon_loss

            g_opt.zero_grad()
            g_loss.backward()
            # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            g_opt.step()

        # Step learning rate scheduler
        # g_sched.step()#(g_loss.item())
        # d_sched.step()#(d_loss.item())

        current_g_lr, current_d_lr = g_opt.param_groups[0]['lr'], d_opt.param_groups[0]['lr']
        tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] D_loss: {d_loss.item():.6f} | G_loss: {g_loss.item():.6f} | Recon: {g_recon_loss.item():.6f} | GLR: {current_g_lr:.6f} | DLR: {current_d_lr:.6f}")

    return generator, discriminator
    
    
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name') # hyperparameter
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes') # hyperparameter
    parser.add_argument('--num_layers', type=int, default=1, help='K, number of top heads to intervene on') # hyperparameter
    parser.add_argument('--alpha', type=float, default=1, help='alpha, intervention strength') # hyperparameter
    parser.add_argument("--num_fold", type=int, default=5, help="number of folds") # hyperparameter
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate') # hyperparameter
    parser.add_argument('--lambda_recon', type=float, default=0, help='Reconstruction loss') # hyperparameter
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--instruction_prompt', default='default', help='instruction prompt for truthfulqa benchmarking, "default" or "informative"', type=str, required=False)
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--scheduler_type', type=str, default='cosine', help='Scheduler type') 
    parser.add_argument('--step_size', type=int, default=5, help='Step size') 
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma')
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset_name == 'sorry-Bench' :
        dataset  = load_dataset("sorry-bench/sorry-bench-202503")['train']
    elif args.dataset_name == 'over-refusal':
        dataset = load_dataset("bench-llm/or-bench", "or-bench-80k")['train']
    elif args.dataset_name == 'malicious-instruct':
        dataset = load_dataset("walledai/MaliciousInstruct")['train']
    elif args.dataset_name == 'advbench':
        dataset  =  load_dataset("walledai/AdvBench")['train']
    elif args.dataset_name == 'jailbreak-bench':
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors") 
        # Please pick one among the available configs: ['behaviors', 'judge_comparison'] 
        dataset  =  ds['harmful'] # dict_keys(['harmful', 'benign'])
    elif args.dataset_name == 'trustllm':
        dataset = load_trust_dataset()
    else: 
        raise ValueError("Invalid dataset name")
    
    # get two folds using numpy
    fold_idxs = np.array_split(np.arange(len(dataset)), args.num_fold)

    # create model
    MODEL = HF_NAMES[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        cache_dir="/home/iplab/LLM/models/",
        attn_implementation="eager").to(DEVICE)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    prompts = formatter(args.model_name, args.dataset_name, dataset, tokenizer)
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    num_key_value_heads = model.config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads

    head_wise_activations = []
    labels = []
    head_wise_activations_base = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_head_wise.npy")
    head_wise_activations_instruct = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}_chat/{args.model_name}_chat_{args.dataset_name}_head_wise.npy")
    labels_base = np.ones(head_wise_activations_base.shape[0])
    labels_instruct = np.zeros(head_wise_activations_instruct.shape[0])

    for i in range(args.num_fold):
        if args.num_fold == 1 :
            train_idxs = fold_idxs[0]
        else :
            train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
            test_idxs = fold_idxs[i]
        head_wise_activations = np.concatenate([head_wise_activations_base[train_idxs], head_wise_activations_instruct[train_idxs]], axis=0)
        labels = np.concatenate([labels_base[train_idxs], labels_instruct[train_idxs]], axis=0)
        perm = np.random.permutation(len(labels))
        head_wise_activations = head_wise_activations[perm]
        labels = labels[perm]
        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(np.arange(len(train_idxs)), size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in np.arange(len(train_idxs)) if x not in train_set_idxs])
        top_layers, _ = get_top_layers(train_set_idxs, val_set_idxs, head_wise_activations, labels, num_layers,
                                        num_heads, args.seed, args.num_layers, args.use_random_dir)

        print("Layers intervened: ", sorted(top_layers))

        interveners = []
        pv_config = []
        for layer in top_layers :
            real_activations = head_wise_activations_base[train_idxs,layer,:]
            fake_activations = head_wise_activations_instruct[train_idxs,layer,:]
            train_dataset = TensorDataset(torch.tensor(real_activations), torch.tensor(fake_activations))
            dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            generator, discriminator = train_adversarial(hidden_size, dataloader, num_epochs=args.num_epochs, lr=args.lr, lambda_recon=args.lambda_recon,
                        scheduler_type=args.scheduler_type, step_size=args.step_size, gamma=args.gamma, device=DEVICE)
            intervener = ISI_Intervener(generator, discriminator, args.alpha) #head=-1 to collect all head activations, multiplier doens't matter
            interveners.append(intervener)
            pv_config.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(intervener),
            })
        intervened_model = pv.IntervenableModel(pv_config, model)

        filename = f'{args.dataset_name}_{args.model_name}_top{args.num_layers}layers_alpha{args.alpha}_advboth'

                                
        alt_evaluate(
            prompts,
            dataset,
            tokenizer,
            test_idxs,
            args.model_name,
            args.dataset_name,
            models={args.model_name: intervened_model},
            output_path=f'/home/iplab/LLM/mitigation_results/responses/{filename}.csv',
            device=DEVICE, 
            instruction_prompt=args.instruction_prompt
        )

        print(f"FOLD {i} Done")
    return filename

if __name__ == "__main__":
    filename = main()
    df_results = pd.concat(df_results, ignore_index=True)
    df_results.to_csv(f'/home/iplab/LLM/mitigation_results/responses/{filename}.csv', index=False)