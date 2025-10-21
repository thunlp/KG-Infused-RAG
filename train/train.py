import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


def preprocessing(example, args, tokenizer):
    prompt = example["prompt"]
    if "llama" in model_args.model_name_or_path.lower():
        template = "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "qwen" in model_args.model_name_or_path.lower():
        template = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    # Ensure the prompt does not exceed the max prompt length
    token_prompt = tokenizer(prompt, truncation=True, max_length=args.max_prompt_length)
    prompt = tokenizer.decode(token_prompt.input_ids)
    prompt = template.format(prompt=prompt)

    if type(example["chosen"]) == dict:
        chosen = example["chosen"]["content"]
        rejected = example["rejected"]["content"]
    else:
        chosen = example["chosen"]
        rejected = example["rejected"]

    one_item = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    return one_item


def main(script_args, training_args, model_args):
    ################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################

    partial_preprocess = partial(preprocessing, args=training_args, tokenizer=tokenizer)
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    if script_args.dataset_name.endswith(".json") or script_args.dataset_name.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=script_args.dataset_name)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = dataset.map(partial_preprocess)

    ##########
    # Training
    ################
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    start_time = time.time()

    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    

    end_time = time.time()
    elapsed_time = int(end_time - start_time)  
    hours, remainder = divmod(elapsed_time, 3600)  
    minutes, seconds = divmod(remainder, 60)  
    print(f"Running Time: {hours} H {minutes} M {seconds} S")


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "dpo", help="Run the DPO training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()

    print("script_args:\n", script_args)
    print("training_args:\n", training_args)
    print("model_args:\n", model_args)

    main(script_args, training_args, model_args)
