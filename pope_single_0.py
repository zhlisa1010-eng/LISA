import sys
import torch
import json
import os
import re
from PIL import Image
from tqdm import tqdm
import argparse

sys.path.insert(0, "/home/user/MLLMs/Qwen2.5vl/transformers")
# sys.path.insert(0, "/home/user/MLLMs/Qwen_VL/model_hf")
import transformers
print("✅ Using transformers from:", transformers.__file__)
from transformers import (
    AutoProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    GenerationConfig,
)
from transformers.generation.utils import GenerationMode
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
def auto_set_answers_file(args):
    subset = "unknown"
    for key in ["adversarial", "random", "popular", "pop+random"]:
        if key in args.question_file:
            subset = key
            break

    decode_suffix = ""
    if args.decode_mode == "greedy":
        decode_suffix = "greedy"
    elif args.decode_mode == "beam":
        decode_suffix = f"beam{args.num_beams}"
    elif args.decode_mode == "sample":
        decode_suffix = f"sample_top{args.top_p}"
    elif args.decode_mode == "dola":
        decode_suffix = f"dola_{args.start_layer}_{args.end_layer}"

    filename = f"qwen2.5_pope_{subset}_{decode_suffix}.jsonl"
    out_dir = "/home/user/MLLMs/Qwen2.5vl/output/POPE"
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)

def recorder(out):
    if isinstance(out, list):  
        out = out[0]
    word_list = re.split(r'[^\w]+', out.lower())
    return "Yes" if "yes" in word_list else "No"

def eval_model(args):
    # device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    # )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype="auto", 
    device_map={"": 0}  
)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    completed_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    completed_ids.add(item["question_id"])
                except:
                    continue

    with open(answers_file, "a", encoding="utf-8") as ans_file:
        for line in tqdm(questions, desc="Generating"):
            idx = line["question_id"]
            if idx in completed_ids:
                continue
            # if idx < 20 :
            #     continue

            image_path = os.path.join(args.image_folder, line["image"])
            question = line["text"]

            try:
                messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(device)
                generation_config = GenerationConfig.from_model_config(model.config)
        
                if args.decode_mode == "greedy":
                        generation_config.do_sample = False
                        generation_config.num_beams = 1
                        generation_config.generation_mode = GenerationMode.GREEDY_SEARCH
                elif args.decode_mode == "beam":
                        generation_config.do_sample = False
                        generation_config.num_beams = 5
                        generation_config.generation_mode = GenerationMode.BEAM_SEARCH
                elif args.decode_mode == "sample":
                        generation_config.do_sample = True
                        generation_config.temperature = max(args.temperature, 1e-5)
                        generation_config.top_p = args.top_p
                        generation_config.num_beams = 1
                        generation_config.generation_mode = GenerationMode.SAMPLE
                elif args.decode_mode == "dola":
                        generation_config.do_sample = False
                        generation_config.num_beams = 1
                        generation_config.dola_layers = list(range(args.start_layer, args.end_layer))
                        generation_config.generation_mode = GenerationMode.DOLA_GENERATION
                else:
                        raise ValueError(f"Unknown decode mode: {args.decode_mode}")
                generated = model.generate(
        **inputs,
        max_new_tokens=512,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_hidden_states=True
)
                sequences = generated.sequences
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, sequences)]
                response= processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                caption = response if isinstance(response, str) else response[0]
                print('*****')
                print(caption)
                yesno = recorder(caption)

                ans_file.write(json.dumps({
                    "question_id": idx,
                    "prompt": question,
                    "text": yesno,
                    "model_id": "qwen-deco",
                    "image": line["image"],
                    "metadata": {}
                }, ensure_ascii=False) + "\n")

                print(f"[{idx}] → {caption} → {yesno}")
                if idx < 10:
                    print(f"[ DEBUG Raw output] {caption}")

            except Exception as e:
                print(f"[Error] {line['image']} failed: {str(e)}")
                continue

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True, help="Path to the POPE question JSONL file")
    parser.add_argument("--image-folder", type=str, required=True, help="Folder path where images are stored")
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=-1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-mode", type=str, required=True, choices=["greedy", "beam", "sample", "dola"])
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for sampling (only used if decode-mode=sample)")
    parser.add_argument("--start-layer", type=int, default=0, help="Start layer for DOLA (only used if decode-mode=dola)")
    parser.add_argument("--end-layer", type=int, default=0, help="End layer for DOLA (only used if decode-mode=dola)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.answers_file == "auto":
        args.answers_file = auto_set_answers_file(args)
        print(f"Auto-generated answers file path: {args.answers_file}")

    eval_model(args)