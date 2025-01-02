import os
import json
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from tqdm import tqdm
from arguments import ModelArguments, DataTrainingArguments, GLMTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from utils import *
from model import GLMModelforIND, LlamaModelForIND, Qwen2ModelForIND
from torch.nn import DataParallel
from dataset import INSTRUCTION, MAGDataset, RawDataset

def calculate_hit_at_k(sorted_parent_scores, true_parents, k):

    top_k_predictions = [score["parent"] for score in sorted_parent_scores[:k]]
    return any(parent in true_parents for parent in top_k_predictions)


def calculate_recall_at_k(sorted_parent_scores, true_parents, k):

    top_k_predictions = [score["parent"] for score in sorted_parent_scores[:k]]
    matched = sum(1 for parent in top_k_predictions if parent in true_parents)
    return matched / len(true_parents)


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GLMTrainingArguments))
model_args, data_args, training_args = parser.parse_json_file(
    json_file="configs/llama3/eval.json"
)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    From https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


set_seed(47)
config = AutoConfig.from_pretrained(
    model_args.model_name_or_path, trust_remote_code=True
)
config.use_cache = False
# config._attn_implementation = "flash_attention_2" #use flash attention
config.model_args = model_args
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path, trust_remote_code=True
)


if training_args.bf16:
    dtype = torch.bfloat16
elif training_args.fp16:
    dtype = torch.float16
else:
    dtype = torch.float32


model = LlamaModelForIND.from_pretrained(
    model_args.model_name_or_path,
    torch_dtype=dtype,
    config=config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
).cuda()


if tokenizer.pad_token is None:
    special_token_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_token_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_token_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_token_dict["unk_token"] = DEFAULT_UNK_TOKEN
smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_token_dict,
    tokenizer=tokenizer,
    model=model,
)
model.add_special_tokens(tokenizer)


if "Llama" in model_args.model_name_or_path or "Qwen2" in model_args.model_name_or_path:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
elif "glm" in model_args.model_name_or_path:
    target_modules = ["query_key_value"]
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=model_args.lora_rank,
    target_modules=target_modules,  # different among different fundation model
    lora_alpha=model_args.lora_alpha,
    lora_dropout=model_args.lora_dropout,
)
model = get_peft_model(model, peft_config).cuda()
if model_args.lora_ckpt_path:  # load lora checkpoint, maybe modified
    if os.path.exists(os.path.join(model_args.lora_ckpt_path, "pytorch_model.bin")):
        paras_path = os.path.join(model_args.lora_ckpt_path, "pytorch_model.bin")
    elif os.path.exists(os.path.join(model_args.lora_ckpt_path, "adapter_model.bin")):
        paras_path = os.path.join(model_args.lora_ckpt_path, "adapter_model.bin")
    else:
        raise ValueError(
            "pytorch_model.bin or adapter_model.bin not found in the lora checkpoint"
        )
    ckpt = torch.load(paras_path)

    for k, v in model.named_parameters():
        if "lora" in k:
            if (
                "default" in k
            ):  # if using torch.save to save peft model, the key will contains "default", such as "base_model.model.model.layers.31.mlp.up_proj.default.weight"
                modify_paras_for_lora = True
            else:  # save using peftmodel.save_pretrained
                modify_paras_for_lora = False
    if modify_paras_for_lora:  # add "default" to the key of the parameters
        modified_ckpt = {}
        for k, v in ckpt.items():
            if "lora" in k and "default" not in k:
                n_list = k.split(".")
                n_list.insert(-1, "default")
                n = ".".join(n_list)
                modified_ckpt[n] = v
            else:
                modified_ckpt[k] = v
        loading_res = model.load_state_dict(modified_ckpt, strict=False)
    else:
        loading_res = model.load_state_dict(ckpt, strict=False)
    assert (
        loading_res.unexpected_keys == []
    ), f"missing keys: {loading_res.missing_keys}"
    model = model.cuda()

model = torch.nn.DataParallel(model)  # 多 GPU 训练
model = model.cuda()

model.eval()


raw_graph_dataset = MAGDataset(
    name="pwc_method",
    path="data/pwc/pwc_method.pickle.bin",
    raw=True,
    existing_partition=False,
)

dataset = RawDataset(
    raw_graph_dataset,
    sampling_mode=1,
    negative_size=15,
    max_pos_size=5,
    expand_factor=40,
    cache_refresh_time=64,
    test_topk=-1,
    tokenizer=tokenizer,
)

graph = dataset.full_graph
pseudo_root_node = dataset.pseudo_root_node
root_nodes = graph.successors(pseudo_root_node)
second_level_nodes = set()
third_level_nodes = set()
# 根节点
for root in root_nodes:
    # 第二层节点
    successors = list(graph.successors(root))
    second_level_nodes.update(successors)
    # 第三层节点
    for node in successors:
        third_level_nodes.update(graph.successors(node))

test_nodes = []
for node in dataset.test_node_list:
    if node in third_level_nodes:
        test_nodes.append(node)

hit_at_1 = 0
hit_at_5 = 0
hit_at_10 = 0
recall_at_1 = 0
recall_at_5 = 0
recall_at_10 = 0

results = []

local_instruct = (
    '\nQuery: "{}"'
    + '\n1. Hypernym Candidate: "{}"\n   Is this a hypernym of the query? Answer: '
    + LABEL_TOKEN
)

input_text = INSTRUCTION.format(local_instruct)

batch_size = 32 * 7

# second_level_nodes 分批
second_level_nodes_batches = [
    list(second_level_nodes)[i : i + batch_size]
    for i in range(0, len(second_level_nodes), batch_size)
]

for test_node in tqdm(test_nodes):
    query = test_node.description
    true_parents = set(graph.predecessors(test_node))

    parent_scores = []

    for batch in second_level_nodes_batches:

        input_texts = [input_text.format(query, node.description) for node in batch]

        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits


        YES_TOKEN_ID, NO_TOKEN_ID = tokenizer.convert_tokens_to_ids(['Yes','No'])
        LABEL_TOKEN_ID = tokenizer.convert_tokens_to_ids(LABEL_TOKEN)
        labels_pos = (inputs["input_ids"] == LABEL_TOKEN_ID).nonzero(as_tuple=False)

        for pos, parent_node in zip(labels_pos, batch):
            batch_idx, mask_idx = pos[0].item(), pos[1].item() - 1
            label_logits = logits[batch_idx, mask_idx, :]

            yes_score = label_logits[YES_TOKEN_ID].item()
            no_score = label_logits[NO_TOKEN_ID].item()

            prediction = "yes" if yes_score > no_score else "no"

            parent_scores.append(
                {
                    "parent": parent_node,
                    "yes_score": yes_score,
                    "no_score": no_score,
                    "score": yes_score - no_score,
                    "prediction": prediction,
                }
            )

    # 降序排列
    sorted_parent_scores = sorted(parent_scores, key=lambda x: x["score"], reverse=True)

    result = {
        "query": test_node.norm_name,
        "true_parents": [parent.norm_name for parent in true_parents],
        "predictions": [item["parent"].norm_name for item in sorted_parent_scores[:10]],
    }
    results.append(result)

    hit_at_1 += calculate_hit_at_k(sorted_parent_scores, true_parents, k=1)
    hit_at_5 += calculate_hit_at_k(sorted_parent_scores, true_parents, k=5)
    hit_at_10 += calculate_hit_at_k(sorted_parent_scores, true_parents, k=10)

    recall_at_1 += calculate_recall_at_k(sorted_parent_scores, true_parents, k=1)
    recall_at_5 += calculate_recall_at_k(sorted_parent_scores, true_parents, k=5)
    recall_at_10 += calculate_recall_at_k(sorted_parent_scores, true_parents, k=10)

    print(f"Query: {query}")
    print(f"True Parents: {', '.join(map(str, true_parents))}")
    for rank, parent_score in enumerate(sorted_parent_scores[:5], start=1):
        print(
            f"Rank {rank}: Parent: {parent_score['parent']}, Score: {parent_score['score']:.4f}"
        )

output_file = "test_result_llama_1227.json"
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

total_samples = len(test_nodes)
print(f"Hit@1: {hit_at_1 / total_samples:.4f}")
print(f"Hit@5: {hit_at_5 / total_samples:.4f}")
print(f"Hit@10: {hit_at_10 / total_samples:.4f}")
print(f"Recall@1: {recall_at_1 / total_samples:.4f}")
print(f"Recall@5: {recall_at_5 / total_samples:.4f}")
print(f"Recall@10: {recall_at_10 / total_samples:.4f}")
