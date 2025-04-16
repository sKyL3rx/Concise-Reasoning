import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from utilities import *
from transformers import AutoTokenizer, AutoModelForCausalLM



target_ids = {2231,2237,2240,2245}
selected_features = get_dataset("Hothan/OlympiadBench", "OE_MM_maths_en_COMP", target_ids)






def get_model():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", torch_dtype = torch.float16)
    model.to('cuda')

    return tokenizer, model

def get_result(tokenizer, model, sample):
    

    prompt = f"""Q:  
    {sample['question']}

    A: Let's think step by step. In the end, put the final answer inside \\boxed{{}}.
    """



    inputs = tokenizer(prompt, return_tensors = 'pt').to(model.device)


    outputs = model.generate(
    **inputs,
    max_new_tokens=15000,
    temperature=0.6,
    top_p=0.95,
    do_sample=True,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response, sample["final_answer"]





if __name__ == "__main__":

    tokenizer, model = get_model()

    for idx, sample in enumerate(selected_features):
        print(f"\n\n\t\t ====================================={idx + 1}===================================")
        res, ground_truth = get_result(tokenizer, model, sample)
        
        print("Model response: ", res)
        print("Real groundtruth: ", ground_truth)

