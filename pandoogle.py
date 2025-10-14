import pandas as pd
import numpy as np
import json
from ollama import embed
import csv
import os
import ollama


json_file_path = "/home/nathan.varner/Documents/dsc360/lab04/pandas_help_corpus.json"
def buildIndex(model):
    embeddings = []
    with open(json_file_path, 'r') as file:
        chunks = json.load(file)
        for item in chunks:
            text = item['doc']
            if text == "":
                continue
            resp = embed(model = model, input = text)
            vec = resp["embeddings"][0]
            embeddings.append(vec)
        embeddings = np.array(embeddings, dtype = np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis = 1, keepdims = True)
        np.save("index/embeddings.npy", embeddings)
        metadata = {
        "model": model,
        "dimension": embeddings.shape[1],
        "normalized": True
        }
        with open("index/metadata.json", "w") as f:
            json.dump(metadata, f, indent = 2)
def search(query: str) -> str:
    with open("index/metadata.json", "r") as f:
        metadata = json.load(f)
    embeddings = np.load("index/embeddings.npy")
    with open(json_file_path, 'r') as file:
        chunks = json.load(file)
    k = 1
    resp = embed(model=metadata["model"], input=query)
    qVector = np.array(resp["embeddings"][0], dtype=np.float32)
    qVector /= np.linalg.norm(qVector)
    scores = np.dot(embeddings, qVector)
    topK = np.argsort(scores)[-k:][::-1]
    answer = chunks[topK[0]]["doc"]
    #print(f"Answer: {answer}\n")
    return query + "||helpful info: " + answer
def query_ollama(prompt: str, model: str) -> str:
    """Call Ollama with the user prompt and return the reply text."""
    try:
        response = ollama.chat(model=model,
                               messages= [{"role": "user",
                                           "content": prompt}],
                                           stream=True)
        final =""
        for chunk in response:
            print(chunk['message']['content'], end='', flush=True)
            final+=chunk['message']['content']
        return final
    except ollama.ResponseError as e:
        print("Error: ", e.error)
def main():
    modelE = "qwen3-embedding:0.6b"
    modelG = "gemma3:1b"
    query = input("Enter your query here: ")
    #buildIndex(modelE)
    query_ollama(search(query), modelG)
if __name__ == "__main__":
    main()
