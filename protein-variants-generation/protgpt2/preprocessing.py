import sys
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch

def read_fasta(filepath, output_arr=False):
    names = []
    seqs = []
    seq = ''
    with open(filepath, 'r') as fin:
        for line in fin:
            if line.startswith('>'):
                if seq:
                    names.append(name)
                    if output_arr:
                        seqs.append(np.array(list(seq)))
                    else:
                        seqs.append(seq)
                name = line[1:].strip()
                seq = ''
            else:
                seq += line.strip()
        if seq:
            names.append(name)
            if output_arr:
                seqs.append(np.array(list(seq)))
            else:
                seqs.append(seq)
    if output_arr:
        seqs = np.array(seqs)
    return names, seqs

def insert_newlines(seq, every=60):
    return '\n'.join(seq[i:i+every] for i in range(0, len(seq), every))

def output_fasta(names, seqs, output_file):
    with open(output_file, 'w') as file:
        for name, seq in zip(names, seqs):
            file.write(name + '\n')
            file.write(seq + '\n')

def main():
    input_file = r"F:\ayush_work\BIO\archive\anatoxin_sequences.fasta"
    preprocessed_file = r"F:\ayush_work\BIO\protein-variants-generation\anatoxin_preprocessed.txt"
    output_file = r"F:\ayush_work\BIO\protein-variants-generation\anatoxin_generated.txt"

    # Preprocess input data
    _, input_seqs = read_fasta(input_file)
    input_seqs = [insert_newlines(seq) for seq in input_seqs]
    input_names = ["<|endoftext|>" for _ in range(len(input_seqs))]
    output_fasta(input_names, input_seqs, preprocessed_file)

    # Use ProtGPT2 model to generate new sequences
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer directly for device control
    model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2")
    tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")

    model.to(device)

    with open(preprocessed_file, 'r') as file:
        sequences = file.readlines()

    num_sequences_to_generate = 300
    sequence_index = 0

    # Open output file in append mode, or create it if it doesn't exist
    with open(output_file, 'w') as file:
        pass  # Just to clear the file at the start

    with open(output_file, 'a') as file:
        for _ in tqdm(range(num_sequences_to_generate)):
            if sequence_index >= len(sequences):
                sequence_index = 0  # Loop back to the start of sequences if needed
            inputs = tokenizer(sequences[sequence_index].strip(), return_tensors='pt').to(device)
            output = model.generate(**inputs, max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=1)
            new_seq = tokenizer.decode(output[0], skip_special_tokens=True)
            file.write(new_seq + '\n')
            file.flush()  # Ensure the line is written immediately
            sequence_index += 1

if __name__ == "__main__":
    main()