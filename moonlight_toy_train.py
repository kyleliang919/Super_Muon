import os
import math
import torch
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
import wandb

class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        if os.path.exists(f"{self.dataset_name}.bin"):
            self.tokens = torch.load(f"{self.dataset_name}.bin")
        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, f"{self.dataset_name}.bin")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py

def get_model_and_dataloader(model_name, dataset_name, hidden_size, batch_size = 16):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    train_dataset = load_dataset(name2path[dataset_name], trust_remote_code=True)
    if model_name == "qwen":
        tokenizer = Qwen2Tokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )
    else:
        assert 0, f"model {model_name} not supported"
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    if model_name == "qwen":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=513,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=16,
            num_hidden_layers=12,
            num_key_value_heads=16,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
        model = Qwen2ForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"
    return model, train_loader


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "mudamw":
        from mudamw import AdamW as MudamW
        return MudamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), muon_exclude = dict([p
            for name, p in model.named_parameters()
            if "embed_tokens" in name or "lm_head" in name])
        )
    elif optimizer_name == "mudamw_orthogonal":
        from mudamw import AdamW as MudamW
        return MudamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), orthogonal_init = True
        )
    elif optimizer_name == "c_mudamw":
        from mudamw import AdamW as MudamW
        return MudamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), cautious = True
        )
    elif optimizer_name == "c_mudamw_orthogonal":
        from mudamw import AdamW as MudamW
        return MudamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), cautious = True, orthogonal_init = True
        )
    elif optimizer_name == "muon":
        from muon import Muon
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    elif optimizer_name == "muon_with_embedding":
        from muon import Muon
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2
            )
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    elif optimizer_name == "sharded_muon":
        from sharded_muon import Muon as Sharded_Muon
        muon_params = [
            (name, p)
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Sharded_Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
            shard_list = {
                "q_proj": {"dim": 0, "num_shards":  model.config.num_attention_heads}, 
                "k_proj": {"dim": 0, "num_shards":  model.config.num_attention_heads}, 
                "v_proj": {"dim": 0, "num_shards":  model.config.num_attention_heads},
                "up_proj": {"dim": 0, "num_shards": 16},
                "down_proj": {"dim": 1, "num_shards": 16},
                "gate_proj": {"dim": 0, "num_shards": 16}
                }
            )
    else:
        assert 0, "optimizer not supported"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--optimizer", type=str, default="adamW")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    run = wandb.init(entity = "kl2", project = "c-optim", name = args.run_name)
    logger.add(f"logs/train_{args.model}_{args.optimizer}_lr{args.lr}.log")
    
    model, train_loader = get_model_and_dataloader(
        args.model, args.dataset, args.hidden_size, args.batch_size
    )
    optimizer = get_optimizer(
        args.optimizer, model, lr=args.lr
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )
    for epoch in range(epoch):
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            input_ids = batch
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            run.log(
                    {"train/step": step,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/loss": loss.item()}
            )
            logger.info(
                f"Epoch: {epoch} Step: {step} LR: {optimizer.param_groups[0]['lr']} Training loss: {loss.item()}"
            )
