import os
import torch
import argparse
from mp_utils import choices, format_example, gen_prompt, softmax, run_eval
from hf_causal_model import eval

from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="../results/not_specified")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--load_in_8bit", action='store_true')
    args = parser.parse_args()

    # TODO: better handle
    tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", device_map="auto", trust_remote_code=True)
    
    state_dict = torch.load(args.model_name_or_path, map_location='cpu') 
    model.load_state_dict(state_dict)  
    run_eval(model, tokenizer, eval, args)