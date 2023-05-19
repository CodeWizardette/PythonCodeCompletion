import openai
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import json
import shutil

import datasets
from datasets import Dataset, load_dataset, concatenate_datasets
from datasets.dataset_dict import DatasetDict
from tqdm import tqdm

import jax
import jax.profiler
import jax.numpy as jnp
import optax
import transformers
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from flax.serialization import to_bytes, from_bytes


def code_suggestion(keyword):
    """Anahtar kelimeye benzer kod ipuçları önerir."""
    codes = [
        "del variable",
        "pass",
        "delattr(object, name)",
        "getattr(object, name)",
        "hasattr(object, name)",
        "setattr(object, name, value)",
        "type(object)",
        "len(sequence)",
        "list(iterable)",
        "tuple(iterable)",
        "dict(**kwargs)",
        "set(iterable)",
        "frozenset(iterable)",
        "max(iterable)",
        "min(iterable)",
        "abs(number)",
        "sum(iterable)",
        "sorted(iterable)",
        "reversed(sequence)",
        "enumerate(iterable)",
        "zip(*iterables)",
        "range(start, stop, step)",
        "round(number, ndigits)",
        "int(x)",
        "float(x)",
        "str(object)",
        "bool(x)",
        "input(prompt)",
        "print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)",
        "open(file, mode='r', encoding='utf-8')",
        "os.path.exists(path)",
        "os.path.isfile(path)",
        "os.path.isdir(path)",
        "os.listdir(path)",
        "os.getcwd()",
        "os.chdir(path)",
        "os.mkdir(path)",
        "os.makedirs(path)",
        "os.remove(path)",
        "os.rename(src, dst)",
        "os.system(command)",
        "shutil.copy(src, dst)",
        "shutil.move(src, dst)",
        "time.sleep(seconds)",
        "math.ceil(x)",
        "math.floor(x)",
        "math.sqrt(x)",
        "math.pow(x, y)",
        "math.sin(x)",
        "math.cos(x)",
        "math.tan(x)",
        "math.log(x)",
        "random.random()",
        "random.randint(a, b)",
        "random.choice(sequence)",
        "random.shuffle(sequence)",
    ]

    if keyword == "def":
        codes.append("def function_name(args):\n\t# Function body\n\tpass")
    elif keyword in ["if", "else", "elif"]:
        codes.append(f"if condition:\n\t# If block\nelse:\n\t# Else block")
    elif keyword == "for":
        codes.append("for item in iterable:\n\t# Loop body")
    elif keyword == "while":
        codes.append("while condition:\n\t# Loop body")
    elif keyword == "try":
        codes.append("try:\n\t# Try block\nexcept Exception as e:\n\t# Exception handling")
    elif keyword == "except":
        codes.append("try:\n\t# Try block\nexcept Exception as e:\n\t# Exception handling")
    elif keyword == "raise":
        codes.append("raise Exception('Error message')")
    elif keyword == "import":
        codes.append("import module")
    elif keyword == "from":
        codes.append("from module import function")
    elif keyword == "as":
        codes.append("import module as alias")
    elif keyword == "return":
        codes.append("return value")
    elif keyword == "yield":
        codes.append("yield value")
    elif keyword == "with":
        codes.append("with open('file.txt', 'r') as file:\n\t# File handling")
    elif keyword == "class":
        codes.append("class ClassName:\n\t# Class body")
    elif keyword == "pass":
        codes.append("pass")
    elif keyword == "assert":
        codes.append("assert condition, 'Error message'")
    elif keyword == "break":
        codes.append("break")
    elif keyword == "continue":
        codes.append("continue")
    elif keyword == "global":
        codes.append("global variable")
    elif keyword == "nonlocal":
        codes.append("nonlocal variable")
    elif keyword == "lambda":
        codes.append("lambda arguments: expression")
    elif keyword == "is":
        codes.append("if variable is None:\n\t# Check if variable is None")
    elif keyword == "in":
        codes.append("if item in list:\n\t# Check if item is in list")
    elif keyword == "not":
        codes.append("if not condition:\n\t# Check if condition is False")
    elif keyword == "and":
        codes.append("if condition1 and condition2:\n\t# Check if both conditions are True")
    elif keyword == "or":
        codes.append("if condition1 or condition2:\n\t# Check if either condition is True")
    elif keyword == "True":
        codes.append("if True:\n\t# True block")
    elif keyword == "False":
        codes.append("if False:\n\t# False block")
    elif keyword == "None":
        codes.append("variable = None")

    return codes


def complete_python_code(prompt, code_lines, max_tokens=100, temperature=0.7):
    """Python kodunu tamamlayan ve döndüren ana fonksiyon."""

    code_prompt = f'```python\n{prompt}\n'
    code_prompt += '\n'.join(code_lines) + '\n```'

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=code_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        stop=None
    )

    completed_text = response.choices[0].text.strip()

    if completed_text.startswith("```python"):
        completed_text = completed_text[len("```python"):].strip()
    if completed_text.endswith("```"):
        completed_text = completed_text[:-len("```")].strip()

    return completed_text


def get_user_input():
    """Kullanıcının kod girişi alınır ve tuple olarak döndürülür."""

    prompt = input("Python kodunu girin:\n")
    code_lines = []
    while True:
        line = input("Sonraki satırı girin (Bırakmak için boş bırakın):\n")
        if not line:
            break
        code_lines.append(line)
    return prompt, code_lines


def main():
    """Programın çalıştırılmasını sağlayan ana fonksiyon."""

    while True:
        prompt, code_lines = get_user_input()

        # Kullanıcının boş bir girdi vermesi durumunda program sonlanır
        if not prompt:
            break

        # Anahtar kelimeye göre ipuçları önerilir
        suggestion = code_suggestion(prompt.split(' ')[0])

        if len(suggestion) > 0:
            print(f"{suggestion[0]} -> Örnek fonksiyon tanımı.")

        completed_code = complete_python_code(prompt, code_lines)
        print("Tamamlanan kod:\n", completed_code)


if __name__ == "__main__":
    main()
