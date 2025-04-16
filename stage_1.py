import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from evaluate import *
from utilities import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead
import tqdm



target_ids = {2231,2237,2240,2245}
selected_features = get_dataset("Hothan/OlympiadBench", "OE_MM_maths_en_COMP", target_ids)







def get_model():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model =  AutoModelForCausalLMWithValueHead.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", torch_dtype = torch.float16)
    tokenizer.pad_token = tokenizer.eos_token
    model.to('cuda')

    return tokenizer, model

def init_ppo(tokenizer, model, dataset):

    
    config = PPOConfig(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
       learning_rate=1.41e-5,
    )



    ppo_trainer = PPOTrainer(
        model = model, 
        config = config,
        dataset = dataset,
        tokenizer = tokenizer

    )

    return ppo_trainer

if __name__ == "__main__":

    def make_prompt(sample):
        return f"""You are an AI assistant. Please solve the following Math competition problem.
                Q:  {sample['question']} \
                 Please provide a step-by-step solution. At the end, give your final boxed answer using \\boxed{{...}}."""
    

    
    tokenizer, model = get_model()

    def tokenize(sample):
        prompt = make_prompt(sample)
        sample["input_ids"] = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids[0]
        return sample

    dataset = selected_features.map(tokenize, batched=False)
    
    generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    }

        

    ppo_trainer = init_ppo(tokenizer, model, dataset)

    epochs = 10

    for epoch in tqdm(range(epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader): 
            query_tensors = batch["input_ids"]

            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

            print(response_tensors)





    

