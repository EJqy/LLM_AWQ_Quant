from datasets import load_dataset

def prepare_calib_tokens(tokenizer, device, num_samples, max_seq_len):
    sample_input_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is your name?"}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(num_samples))
    ds = ds.map(
        lambda batch: {
            "text": tokenizer.apply_chat_template(batch["messages"], tokenize=False)
        }
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    calibration_tokens = tokenizer(
        ds["text"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
        add_special_tokens=False,
    ).input_ids.to(device)
    return calibration_tokens
    # print("Calibration tokens:", calibration_tokens.shape)