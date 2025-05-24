import torch
from einops import rearrange
import numpy as np
import pickle
import os, json
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

result_df = []
import pyvene as pv
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_NAMES = {
    'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama3_8B': "meta-llama/Llama-3.1-8B",
    'opt6.7B': "facebook/opt-6.7b",
    'Qwen2.5_7B' : "Qwen/Qwen2.5-7B",
    'Yi1.5_9B' : "01-ai/Yi-1.5-9B",
    'Ministral_8B' : "mistralai/Ministral-8B-Instruct-2410",
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
        if dataset_name in ['truthfulqa', 'triviaqa']:
            prompt = tokenizer(base_prompt.format(question=row['question']), return_tensors='pt')
        elif dataset_name == 'tydiqa':
            prompt = tokenizer(base_prompt.format(context=row['context'], question=row['question']), return_tensors='pt')
        elif dataset_name == 'coqa':
            prompt = tokenizer(base_prompt.format(story=row['story'], question=row['question']), return_tensors='pt')
        elif dataset_name == 'haluevaldia':
            prompt = tokenizer(base_prompt.format(knowledge=row['knowledge'], dialogue_history=row['dialogue_history']), return_tensors='pt')
        elif dataset_name == 'haluevalqa':
            prompt = tokenizer(base_prompt.format(context=row['knowledge'], question=row['question']), return_tensors='pt')
        elif dataset_name == 'haluevalsum':
            prompt = tokenizer(base_prompt.format(document=row['document']), return_tensors='pt')
        all_prompts.append(prompt)
        
    return all_prompts

def run_answers(test_idxs, model_name, dataset_name, questions, dataset, 
                tokenizer=None,  model=None, verbose=False, device=DEVICE, instruction_prompt="default"):
    frame = dataset.to_pandas()
    # tokens = formatter(model_name, dataset_name, dataset, tokenizer)
    print(questions[0])
    sequences = []
    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(questions, desc="run_answers")):
            if idx not in test_idxs:
                continue
            # max_len = inputs.input_ids.shape[-1] + 64

            # --- intervention code --- #
            inputs = inputs.to(device)
            _, output = model.generate(inputs, top_k=1, max_new_tokens=64, num_return_sequences=1,)
            # output = model.generate(**inputs, top_k=1, max_length=max_len, num_return_sequences=1,)
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
        
    return frame.dropna(subset=['intervened_answers'])

def alt_evaluate(questions, dataset, tokenizer, test_idxs,model_name,dataset_name, models, output_path, device=DEVICE,
                    verbose=False, instruction_prompt="default"):    
    for mdl in models.keys(): 

        llama_model = models[mdl]
        answers = run_answers(test_idxs, model_name, dataset_name, questions, dataset, tokenizer, model=llama_model,
                        device=device, verbose=verbose,
                        instruction_prompt=instruction_prompt)

        result_df.append(answers)#.to_csv(output_path)

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
 
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default=None, help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
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

    # for model_name, dataset_name in [("llama2_7B", "malicious-instruct"), 
    #                                  ("llama2_7B_chat", "malicious-instruct"),
    #                                  ("llama2_7B", "advbench"), 
    #                                  ("llama2_7B_chat", "advbench"), 
    #                                  ("llama2_7B", "jailbreak-bench"),
    #                                  ("llama2_7B_chat", "jailbreak-bench"), 
    #                                  #("llama2_7B", "over-refusal"), 
    #                                  # ("llama2_7B_chat", "over-refusal"),
    #                                  ("llama2_7B", "trustllm"), 
    #                                  ("llama2_7B_chat", "trustllm")]:
    # for model_name, dataset_name in [("llama2_7B", "malicious-instruct_reject_no"), 
    #                                  ("llama2_7B", "malicious-instruct_reject_yes"),
    #                                  ("llama2_7B", "advbench_reject_no"), 
    #                                  ("llama2_7B", "advbench_reject_yes"), 
    #                                  ("llama2_7B", "jailbreak-bench_reject_no"),
    #                                  ("llama2_7B", "jailbreak-bench_reject_yes"), 
    #                                  #("llama2_7B", "over-refusal"), 
    #                                  # ("llama2_7B_chat", "over-refusal"),
    #                                  ("llama2_7B", "trustllm_reject_no"), 
    #                                  ("llama2_7B", "trustllm_reject_yes")]:
    # for model_name, dataset_name in [("llama3_8b/llama3_8B", "malicious-instruct_reject_no"), 
    #                                  ("llama3_8b/llama3_8B", "malicious-instruct_reject_yes"),]:
                                    #  ("llama3_8b/llama3_8B", "advbench_reject_no"), 
                                    #  ("llama3_8b/llama3_8B", "advbench_reject_yes"), 
                                    #  ("llama3_8b/llama3_8B", "jailbreak-bench_reject_no"),
                                    #  ("llama3_8b/llama3_8B", "jailbreak-bench_reject_yes"), 
                                    #  #("llama3_8B", "over-refusal"), 
                                    #  # ("llama3_8B", "over-refusal"),
                                    #  ("llama3_8b/llama3_8B", "trustllm_reject_no"), 
                                    #  ("llama3_8b/llama3_8B", "trustllm_reject_yes")]:
        
        # head_wise_activations.append(np.load(f"/home/iplab/LLM/mitigation_results/{model_name}_{dataset_name}_head_wise.npy"))
        # labels.append(np.load(f"/home/iplab/LLM/mitigation_results/{model_name}_{dataset_name}_labels.npy"))
    head_wise_activations_hal = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_head_wise.npy")
    head_wise_activations_truth = np.load(f"/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_head_wise_truthful.npy")
    labels_hal = np.zeros(head_wise_activations_hal.shape[0])
    labels_nhal = np.ones(head_wise_activations_truth.shape[0])
    
    head_wise_activations_hal = rearrange(head_wise_activations_hal, 'b l (h d) -> b l h d', h = num_heads)
    head_wise_activations_truth = rearrange(head_wise_activations_truth, 'b l (h d) -> b l h d', h = num_heads)

    # tuning dataset: no labels used, just to get std of activations along the direction
    # activations_dataset = args.dataset_name if args.activations_dataset is None else args.activations_dataset
    # tuning_activations = np.load(f"../features/{args.model_name}_{activations_dataset}_head_wise.npy")
    # tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
    # tuning_labels = np.load(f"../features/{args.model_name}_{activations_dataset}_labels.npy")
    

    # separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)
    # run k-fold cross validation
    for i in range(args.num_fold):
        
        if args.num_fold == 1:
            train_idxs = fold_idxs[0]
            test_idxs = fold_idxs[0]
        else:
            train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
            test_idxs = fold_idxs[i]
        head_wise_activations = np.concatenate([head_wise_activations_hal[train_idxs], head_wise_activations_truth[train_idxs]], axis=0)
        labels = np.concatenate([labels_hal[train_idxs], labels_nhal[train_idxs]], axis=0)
        perm = np.random.permutation(len(train_idxs))
        head_wise_activations = head_wise_activations[perm]
        tuning_activations = head_wise_activations.copy()
        labels = labels[perm]
        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(np.arange(len(train_idxs)), size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in np.arange(len(train_idxs)) if x not in train_set_idxs])

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

        # if args.use_center_of_mass:
        #     filename += '_com'
        # if args.use_random_dir:
        #     filename += '_random'
            
                                
        alt_evaluate(
            prompts,
            dataset,
            tokenizer,
            test_idxs,
            args.model_name,
            args.dataset_name,
            models={args.model_name: intervened_model},
            output_path=f'/home/iplab/LLM/mitigation_results/responses_hal/{filename}.csv',
            device=DEVICE, 
            instruction_prompt=args.instruction_prompt
        )

        print(f"FOLD {i} Done")
    return filename


if __name__ == "__main__":
    filename = main()
    result_df = pd.concat(result_df, ignore_index=True)
    result_df.to_csv(f'/home/iplab/LLM/mitigation_results/responses_hal/{filename}.csv', index=False)