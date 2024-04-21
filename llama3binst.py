import transformers
import torch
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if end_time - start_time < 60:
            print(f"Function '{func.__name__}' took {end_time - start_time} seconds to execute.")
        else:
            print(f"Function '{func.__name__}' took {(end_time - start_time)/60} minutes to execute.")
        return result
    return wrapper

@measure_time
def runme_i():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cpu",
    )

    messages = [
        {"role": "system", "content": "You are a professor chatbot who always responds!"},
        {"role": "user", "content": "what is capital city of South Kore? and who is the prsident of the country"},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"][len(prompt):])

runme_i()


