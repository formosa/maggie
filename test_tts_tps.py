<<<<<<< HEAD
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def measure_tts_tokens_per_second(model_name="mistralai/Mixtral-7B-Instruct-v0.3", input_text="Real-time speech is amazing!"):
    """
    Measure tokens-per-second for a given model on your RTX 3080.

    Parameters
    ----------
    model_name : str
        Name of the model to load from Hugging Face.
    input_text : str
        Sample text to process.

    Returns
    -------
    float
        Tokens per second based on inference time.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)  # Simulate TTS intermediate step
    end_time = time.time()

    tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
    time_taken = end_time - start_time
    return tokens_generated / time_taken

if __name__ == "__main__":
    tokens_per_sec = measure_tts_tokens_per_second()
=======
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def measure_tts_tokens_per_second(model_name="mistralai/Mixtral-7B-Instruct-v0.3", input_text="Real-time speech is amazing!"):
    """
    Measure tokens-per-second for a given model on your RTX 3080.

    Parameters
    ----------
    model_name : str
        Name of the model to load from Hugging Face.
    input_text : str
        Sample text to process.

    Returns
    -------
    float
        Tokens per second based on inference time.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)  # Simulate TTS intermediate step
    end_time = time.time()

    tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
    time_taken = end_time - start_time
    return tokens_generated / time_taken

if __name__ == "__main__":
    tokens_per_sec = measure_tts_tokens_per_second()
>>>>>>> 6062514b96de23fbf6dcdbfd4420d6e2f22903ff
    print(f"Tokens per second: {tokens_per_sec:.2f}")