from transformers import AutoConfig, AutoTokenizer
import argparse
import torch

from ..src.wrappers import RetrievalGenerator, RetrievalGeneratorGPT2, RetrievalGeneratorRoberta


'''
Generation script for SUREALM online generation.
'''


def inference(tokenizer, model, prompts="", device=torch.device("cpu"), **kwargs):
    outputs = []
    sos = tokenizer.bos_token + " "
    if isinstance(prompts, list):
        for i in range(len(prompts)):
            prompts[i] = sos+ prompts[i]
    else:
    prompts = [sos + prompts]  # Currently doesn't support batched decoding
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).input_ids
        print(input_ids)
        input_ids = input_ids.to(device)
        outputs.extend(model.generate(inputs=input_ids, **kwargs))
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
        

def inference_init(args=None, model_path=None, sbert_path=None, knowledge_path=None, max_length=None, max_context_length_per_k=None, **kwargs):
    if args is not None:
        model_path = args.model_path
        sbert_path = args.sbert_path
        knowledge_path = args.knowledge_path
        max_length = args.max_length
        max_context_length_per_k = args.max_context_length_per_k
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if args.model_type == "gpt2"
        model = RetrievalGeneratorGPT2.from_pretrained(model_path, tokenizer=tokenizer)
    elif args.model_type == "bert":
        model = RetrievalGenerator.from_pretrained(model_path, tokenizer=tokenizer)
    else:
        model = RetrievalGeneratorRoberta.from_pretrained(model_path, tokenizer=tokenizer)
    model.post_post_init(max_length=max_length, max_context_length_per_k=max_context_length_per_k)
    model.load_components(sbert_path=sbert_path, knowledge_path=knowledge_path)
    model.to(args.device)
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sbert_path", type=str, default="multi-qa-MiniLM-L6-cos-v1")
    parser.add_argument("--knowledge_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_context_length_per_k", type=int, default=None)
    parser.add_argument("--interactive", action="store_true", default=False)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--num_beams", type=int, default=10)
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    model, tok = inference_init(args)
    tok.bos_token_id = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    tok.eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
    tok.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    assert not (args.interactive and args.prompt_file), f"cannot specify prompt file while interactive mode is enabled."
    assert args.interactive or args.prompt_file, f"specify prompt file or enable interactive mode."
    if args.interactive:
        while True:
            prompt = input("input prompt, type \"!quit\" to exit: ")
            if prompt == "!quit":
                exit()
            print(inference(tok, model, prompt, device=args.device, bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, retrieval_stride=args.stride, retrieval_topk=args.topk))
    else:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f]
            print(inference(tok, model, prompts, device=args.device, bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, retrieval_stride=args.stride, retrieval_topk=args.topk))

if __name__ == "__main__":
    main()