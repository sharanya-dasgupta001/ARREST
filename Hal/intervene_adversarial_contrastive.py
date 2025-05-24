import torch
from einops import rearrange
import numpy as np
import pickle
import os, json
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pyvene as pv

df_results = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_NAMES = {
    'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama3_8B': "meta-llama/Llama-3.1-8B",
    'opt6.7B': "facebook/opt-6.7b",
    'llama2_7B_chat' : "meta-llama/Llama-2-7b-chat-hf"
}

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
        
    for index, row in enumerate(dataset):
        if dataset_name in ['truthfulqa', 'triviaqa']:
            prompt = tokenizer(base_prompt.format(question=row['question']), return_tensors='pt').input_ids
        elif dataset_name == 'tydiqa':
            prompt = tokenizer(base_prompt.format(context=row['context'], question=row['question']), return_tensors='pt').input_ids
        elif dataset_name == 'coqa':
            prompt = tokenizer(base_prompt.format(story=row['story'], question=row['question']), return_tensors='pt').input_ids
        elif dataset_name == 'haluevaldia':
            prompt = tokenizer(base_prompt.format(knowledge=row['knowledge'], dialogue_history=row['dialogue_history']), return_tensors='pt').input_ids
        elif dataset_name == 'haluevalqa':
            prompt = tokenizer(base_prompt.format(context=row['knowledge'], question=row['question']), return_tensors='pt').input_ids
        elif dataset_name == 'haluevalsum':
            prompt = tokenizer(base_prompt.format(document=row['document']), return_tensors='pt').input_ids
            
        all_prompts.append(prompt)
        
    return all_prompts

def run_answers(test_idxs, model_name, dataset_name, questions, dataset,  model=None,
                tokenizer=None, verbose=False, device=DEVICE, instruction_prompt="default"):
    frame = dataset.to_pandas()
    # tokens = formatter(model_name, dataset_name, dataset, tokenizer)

    sequences = []
    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(questions, desc="run_answers")):
            # if idx not in test_idxs:
            #     continue
            max_len = inputs.input_ids.shape[-1] + 64

            # --- intervention code --- #
            inputs = inputs.to(device)
            _, output = model.generate(inputs, top_k=1, max_new_tokens=64, num_return_sequences=1,)
            # output = model.generate(input_ids, top_k=1, max_length=max_len, num_return_sequences=1,)

            model_gen_tokens = output[:, inputs.input_ids.shape[-1]:]
            model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
            model_gen_str = model_gen_str.strip()

            # try: 
            #     # remove everything after 'Q:'
            #     model_gen_str = model_gen_str.split("Q:")[0].strip()
            #     # keep everything after A: 
            #     model_gen_str = model_gen_str.split("A:")[1].strip()
            # except: 
                # pass

            if verbose: 
                print("MODEL_OUTPUT: ", model_gen_str)
            
            frame.loc[idx, 'intervened_answers'] = model_gen_str
            sequences.append(model_gen_str)

            # --- intervention code --- #

    if device:
        torch.cuda.empty_cache()
    
    if dataset_name in ['truthfulqa', 'triviaqa', 'coqa', 'haluevalqa']:
        frame = frame[['question', 'intervened_answers']].rename(columns={'question': 'prompt'})
    elif dataset_name == 'tydiqa' :
        frame = frame[['context', 'intervened_answers']].rename(columns={'context': 'prompt'})
    elif dataset_name == 'haluevaldia' :
        frame = frame[['dialogue_history', 'intervened_answers']].rename(columns={'dialogue_history': 'prompt'})
    elif dataset_name == 'haluevalsum' :
        frame = frame[['document', 'intervened_answers']].rename(columns={'document': 'prompt'})

def alt_evaluate(questions, dataset, tokenizer, test_idxs, model_name, dataset_name, models, output_path, device=DEVICE,
                    verbose=False, instruction_prompt="default"): 

    for mdl in models.keys(): 
        llama_model = models[mdl]
        questions = run_answers(test_idxs, model_name, dataset_name, questions, dataset, model=llama_model, tokenizer=tokenizer,
                        device=device, verbose=verbose,
                        instruction_prompt=instruction_prompt)

        # questions.to_csv(output_path)
        df_results.append(questions)

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

def train_probes(seed, train_set_idxs, val_set_idxs, head_wise_activations, labels, num_layers, num_heads):
    
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

def train_probes2(seed, train_set_idxs, val_set_idxs, head_wise_activations, labels, num_layers, num_heads):

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
    # all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)

    top_heads = []

    top_accs = np.argsort(all_head_accs_np.reshape(num_layers))[::-1][:num_to_intervene]
    # top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    # if use_random_dir: 
    #     # overwrite top heads with random heads, no replacement
    #     random_idxs = np.random.choice(num_heads*num_layers, num_heads*num_layers, replace=False)
    #     top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs[:num_to_intervene]]

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
    
    
def contrastive_loss(anchor, positive, negative, margin=1.0):
    # L2 normalize
    anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
    positive = torch.nn.functional.normalize(positive, p=2, dim=1)
    negative = torch.nn.functional.normalize(negative, p=2, dim=1)
    # Cosine similarity
    pos_sim = torch.nn.functional.cosine_similarity(anchor, positive)
    neg_sim = torch.nn.functional.cosine_similarity(anchor, negative)
    # Contrastive loss (margin ranking loss)
    loss = torch.clamp(margin - pos_sim + neg_sim, min=0.0)
    return loss.mean()

def train_adversarial(HIDDEN_DIM, dataloader, num_epochs=50, lr = 1e-5, lambda_recon=1e-5, lambda_contrastive =  0.5,
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
        for base_batch, chat_batch, reject_batch, non_reject_batch in dataloader:
            base_batch = base_batch.to(device)
            chat_batch = chat_batch.to(device)
            reject_batch = reject_batch.to(device)
            non_reject_batch = non_reject_batch.to(device)
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
            g_contrastive_loss = contrastive_loss(fake_batch, reject_batch, non_reject_batch)
            # torch.clamp(mse(fake_batch, chat_batch), max=1.0)
            g_loss = g_adv_loss + lambda_recon * g_recon_loss + lambda_contrastive * g_contrastive_loss

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
    parser.add_argument('--model_name', type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default=None, help='feature bank for calculating std along direction')
    parser.add_argument('--num_layers', type=int, default=1, help='K, number of top layers to intervene on')
    parser.add_argument('--alpha', type=float, default=1, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=5, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--instruction_prompt', default='default', help='instruction prompt for truthfulqa benchmarking, "default" or "informative"', type=str, required=False)
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lambda_recon', type=float, default=0, help='Reconstruction loss')
    parser.add_argument('--scheduler_type', type=str, default='cosine', help='Scheduler type')
    parser.add_argument('--step_size', type=int, default=5, help='Step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma')
    parser.add_argument('--lambda_contrastive', type=float, default=0.5, help='Contrastive loss')
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    elif args.dataset_name == 'tydiqa':
        dataset = load_dataset("tydiqa", "secondary_task", split="train")
        dataset = dataset.filter(lambda row: "english" in row["id"])
    elif args.dataset_name == 'coqa':
        dataset =  load_coqa_dataset()
    elif args.dataset_name == 'haluevaldia':
        dataset = load_dataset("pminervini/HaluEval", "dialogue")['data']
    elif args.dataset_name == 'haluevalqa':
        dataset = load_dataset("pminervini/HaluEval", "qa")['data']
    elif args.dataset_name == 'haluevalsum':
        dataset = load_dataset("pminervini/HaluEval", "summarization")['data']
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

    # load activations 
    # for model_name, dataset_name in [("llama2_7B", "malicious-instruct"), ("llama2_7B_chat", "malicious-instruct"),
    #     ("llama2_7B", "advbench"), ("llama2_7B_chat", "advbench"), ("llama2_7B", "jailbreak-bench"),
    #     ("llama2_7B_chat", "jailbreak-bench"), #("llama2_7B", "over-refusal"), ("llama2_7B_chat", "over-refusal"),
    #     ("llama2_7B", "trustllm"), ("llama2_7B_chat", "trustllm")]:
    #     head_wise_activations.append(np.load(f"/home/iplab/LLM/mitigation_results/{model_name}_{dataset_name}_head_wise.npy"))
    #     labels.append(np.load(f"/home/iplab/LLM/mitigation_results/{model_name}_{dataset_name}_labels.npy"))
    # head_wise_activations = np.concatenate(head_wise_activations, axis=0)
    # labels = np.concatenate(labels, axis=0)
    head_wise_activations_hal = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_head_wise.npy")
    head_wise_activations_instruct = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}_instruct/{args.model_name}_instruct_{args.dataset_name}_head_wise.npy")
    # labels_base = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_labels.npy")
    # labels_instruct = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}_instruct/{args.model_name}_instruct_{args.dataset_name}_labels.npy")
    labels_base = np.ones(head_wise_activations_hal.shape[0])
    labels_instruct = np.zeros(head_wise_activations_instruct.shape[0])
    rejected_activations = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_reject_yes_head_wise.npy")
    non_rejected_activations = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_reject_no_head_wise.npy")
    
    # head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

    # tuning dataset: no labels used, just to get std of activations along the direction
    # activations_dataset = args.dataset_name if args.activations_dataset is None else args.activations_dataset
    # tuning_activations = np.load(f"../features/{args.model_name}_{activations_dataset}_head_wise.npy")
    # tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
    # tuning_labels = np.load(f"../features/{args.model_name}_{activations_dataset}_labels.npy")
    # tuning_activations = head_wise_activations.copy()

    # separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)
    # run k-fold cross validation
    for i in range(args.num_fold):
        if args.num_fold == 1 :
            train_idxs = fold_idxs[0]
        else :
            train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
            test_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        # save train and test splits
        # df = dataset.to_pandas()
        # os.makedirs(f"splits", exist_ok=True)
        # df.iloc[train_set_idxs].to_csv(f"splits/fold_{i}_train_seed_{args.seed}.csv", index=False)
        # df.iloc[val_set_idxs].to_csv(f"splits/fold_{i}_val_seed_{args.seed}.csv", index=False)
        # df.iloc[test_idxs].to_csv(f"splits/fold_{i}_test_seed_{args.seed}.csv", index=False)

        # get directions
        # if args.use_center_of_mass:
        #     com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, head_wise_activations, labels)
        # else:
        #     com_directions = None
        head_wise_activations_training = np.concatenate([head_wise_activations_hal[train_idxs,:,:], head_wise_activations_instruct[train_idxs,:,:]], axis=0)
        labels_training = np.concatenate([labels_base[train_idxs], labels_instruct[train_idxs]], axis=0)
        perm = np.random.permutation(len(labels_training))
        head_wise_activations_training = head_wise_activations_training[perm]
        labels_training = labels_training[perm]
        top_layers, _ = get_top_layers(train_set_idxs, val_set_idxs, head_wise_activations_training, labels_training, num_layers,
                                        num_heads, args.seed, args.num_layers, args.use_random_dir)
        
        print("Layers intervened: ", sorted(top_layers))

        interveners = []
        pv_config = []
        for layer in top_layers :
            real_activations = head_wise_activations_instruct[train_idxs,layer,:]
            fake_activations = head_wise_activations_hal[train_idxs,layer,:]
            positive_activations = rejected_activations[train_idxs,layer,:]
            negative_activations = non_rejected_activations[train_idxs,layer,:]
            train_dataset = TensorDataset(torch.tensor(fake_activations), torch.tensor(real_activations), torch.tensor(positive_activations), torch.tensor(negative_activations))
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

        filename = f'{args.dataset_name}_{args.model_name}_top{args.num_layers}layers_alpha{args.alpha}_advcontr_attn_fold{i}'

                                
        alt_evaluate(
            dataset,
            test_idxs,
            args.model_name,
            args.dataset_name,
            models={args.model_name: intervened_model},
            output_path=f'/home/iplab/LLM/mitigation_results/responses_hal/{filename}.csv',
            device=DEVICE, 
            instruction_prompt=args.instruction_prompt
        )

        print(f"FOLD {i} Done")


if __name__ == "__main__":
    main()