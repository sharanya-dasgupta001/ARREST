import torch
from einops import rearrange
import numpy as np
import pandas as pd
import os, json
from tqdm import tqdm
import numpy as np
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pyvene as pv
df_results = []

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_NAMES = {
    'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama3_8B': "meta-llama/Llama-3.1-8B",
    'opt6.7B': "facebook/opt-6.7b",
    'Qwen2.5_7B' : "Qwen/Qwen2.5-7B", 
    'Yi1.5_9B' : "01-ai/Yi-1.5-9B", 
    'vicuna_7B' : "lmsys/vicuna-7b-v1.5"
}

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
        if dataset_name == 'sorry-Bench' :
            prompt = tokenizer(base_prompt.format(prompt=row['turns'][0]), return_tensors='pt')
        elif dataset_name in ['over-refusal','malicious-instruct','advbench','trustllm']:
            prompt = tokenizer(base_prompt.format(prompt=row['prompt']), return_tensors='pt')
        elif dataset_name == 'jailbreak-bench':
            prompt = tokenizer(base_prompt.format(prompt=row['Goal']),return_tensors='pt')
            
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
            _,output = model.generate(inputs, top_k=1, max_new_tokens=64, num_return_sequences=1,)

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

def train_probes(seed, train_set_idxs, val_set_idxs, head_wise_activations,
                labels, num_layers, num_heads):
    
    all_head_accs = []
    probes = []

    all_X_train = head_wise_activations[train_set_idxs]
    all_X_val = head_wise_activations[val_set_idxs]
    y_train = labels[train_set_idxs]
    y_val = labels[val_set_idxs]

    for layer in tqdm(range(num_layers), desc="train_probes"): 
        for head in range(num_heads): 
            X_train = all_X_train[:,layer,head,:]
            X_val = all_X_val[:,layer,head,:]
    
            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)

    all_head_accs_np = np.array(all_head_accs)

    return probes, all_head_accs_np

def train_probes2(seed, train_set_idxs, val_set_idxs, head_wise_activations, labels,
                num_layers, num_heads):

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

def get_top_heads(train_idxs, val_idxs, activations, labels, num_layers, num_heads,
                seed, num_to_intervene, use_random_dir=False):
    
    probes, all_head_accs_np = train_probes2(seed, train_idxs, val_idxs, activations, labels,
                                            num_layers=num_layers, num_heads=num_heads)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)

    top_heads = []

    top_accs = np.argsort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    if use_random_dir: 
        # overwrite top heads with random heads, no replacement
        random_idxs = np.random.choice(num_heads*num_layers, num_heads*num_layers, replace=False)
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs[:num_to_intervene]]

    return top_heads, probes

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class ITI_Intervener():
    collect_state = True
    collect_action = True
    attr_idx = -1
    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        action = self.direction.to(b.device)
        self.actions.append(action.detach().clone())
        b[0, -1] = b[0, -1] + action * self.multiplier
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
    
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on') # hyper-parameter
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength') # hyper-parameter
    parser.add_argument("--num_fold", type=int, default=5, help="number of folds") 
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--instruction_prompt', default='default', help='instruction prompt for truthfulqa benchmarking, "default" or "informative"', type=str, required=False)
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

    head_wise_activations_base = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_head_wise.npy")
    head_wise_activations_instruct = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}_chat/{args.model_name}_chat_{args.dataset_name}_head_wise.npy")
    labels_base = np.zeros(len(head_wise_activations_base))
    labels_instruct = np.ones(len(head_wise_activations_instruct))
    
    head_wise_activations_base = rearrange(head_wise_activations_base, 'b l (h d) -> b l h d', h = num_heads)
    head_wise_activations_instruct = rearrange(head_wise_activations_instruct, 'b l (h d) -> b l h d', h = num_heads)

    for i in range(args.num_fold):
        
        if args.num_fold == 1:
            train_idxs = fold_idxs[0]
            test_idxs = fold_idxs[0]
        else:
            train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
            test_idxs = fold_idxs[i]
        head_wise_activations = np.concatenate([head_wise_activations_base[train_idxs], head_wise_activations_instruct[train_idxs]], axis=0)
        labels = np.concatenate([labels_base[train_idxs], labels_instruct[train_idxs]], axis=0)
        perm = np.random.permutation(len(train_idxs))
        head_wise_activations = head_wise_activations[perm]
        tuning_activations = head_wise_activations.copy()
        labels = labels[perm]
        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(np.arange(len(train_idxs)), size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in np.arange(len(train_idxs)) if x not in train_set_idxs])
        top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, head_wise_activations, labels, num_layers,
                                        num_heads, args.seed, args.num_heads, args.use_random_dir)

        print("Heads intervened: ", sorted(top_heads))

        interveners = []
        pv_config = []
        top_heads_by_layer = {}
        for layer, head, in top_heads:
            if layer not in top_heads_by_layer:
                top_heads_by_layer[layer] = []
            top_heads_by_layer[layer].append(head)
        for layer, heads in top_heads_by_layer.items():
            direction = torch.zeros(head_dim * num_heads).to("cpu")
            for head in heads:
                dir = torch.tensor(probes[layer_head_to_flattened_idx(layer, head, num_heads)], dtype=torch.float32).to("cpu")
                dir = dir / torch.norm(dir)
                # print(dir.shape)
                activations = torch.tensor(tuning_activations[:,layer,head,:], dtype=torch.float32).to("cpu") # batch x 128
                proj_vals = activations @ dir.T
                proj_val_std = torch.std(proj_vals)
                direction[head * head_dim: (head + 1) * head_dim] = dir * proj_val_std
            intervener = ITI_Intervener(direction, args.alpha) #head=-1 to collect all head activations, multiplier doens't matter
            interveners.append(intervener)
            pv_config.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(intervener),
            })
        intervened_model = pv.IntervenableModel(pv_config, model) 

        filename = f'{args.dataset_name}_{args.model_name}_top{args.num_heads}heads_alpha{args.alpha}_iti'
            
                                
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