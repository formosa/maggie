# C:\AI\claude\service\maggie\testing\mistral_7b_prompt_example_torch.py

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def prompt_mistral_7b_torch(prompt: str, model_path: str = r"C:\AI\claude\service\maggie\maggie\models\llm\mistral-7b-instruct-v0.3-GPTQ-4bit"):
    """
    Prompts the Mistral 7B Instruct model using torch and transformers, and returns the generated text.

    Parameters
    ----------
    prompt : str
        The input prompt for the language model.
    model_path : str, optional
        The file path to the Mistral 7B GPTQ model directory.
        Defaults to 'C:\AI\claude\service\maggie\maggie\models\llm\mistral-7b-instruct-v0.3-GPTQ-4bit'.

    Returns
    -------
    str
        The generated text from the language model.
    """
    start_time = time.perf_counter()

    # Ensure the model path exists.
    if not os.path.exists(model_path):
        return f"Error: Model path '{model_path}' does not exist."

    try:
        # Load tokenizer and model.
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True) # add use_fast=True
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Move model to GPU if available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model_load_time = time.perf_counter() - start_time

        start_generation = time.perf_counter()

        # Tokenize input prompt.
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate response.
        outputs = model.generate(**inputs, max_new_tokens=256)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        generation_time = time.perf_counter() - start_generation

        total_time = time.perf_counter() - start_time

        print(f"Model Load Time: {model_load_time:.4f} seconds")
        print(f"Generation Time: {generation_time:.4f} seconds")
        print(f"Total Processing Time: {total_time:.4f} seconds")

        return response

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    response = prompt_mistral_7b_torch(user_prompt)
    print("Response:")
    print(response)