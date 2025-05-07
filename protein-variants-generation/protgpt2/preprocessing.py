import os
import glob
from tqdm import tqdm
from transformers import pipeline
import torch

def preprocess_fasta(input_file):
    """Read FASTA file and return sequences with ProtGPT2 formatting"""
    sequences = []
    with open(input_file, 'r') as f:
        current_seq = []
        for line in f:
            if line.startswith('>'):
                if current_seq:
                    sequences.append(format_for_protgpt(''.join(current_seq)))
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_seq:
            sequences.append(format_for_protgpt(''.join(current_seq)))
    return sequences

def format_for_protgpt(sequence, line_length=60):
    """Format sequence for ProtGPT2 input"""
    formatted = []
    for i in range(0, len(sequence), line_length):
        formatted.append(sequence[i:i+line_length])
    return '<|endoftext|>\n' + '\n'.join(formatted)

def generate_sequences(model, tokenizer, num_sequences, device):
    """Generate new sequences using ProtGPT2"""
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=device,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated = []
    for _ in tqdm(range(num_sequences), desc="Generating sequences"):
        output = generator(
            '<|endoftext|>',
            max_length=100,
            do_sample=True,
            top_k=950,
            repetition_penalty=1.2,
            num_return_sequences=1
        )
        seq = output[0]['generated_text'].split('<|endoftext|>')[-1].strip()
        generated.append(seq.replace('\n', ''))  # Remove formatting newlines
    return generated

def process_dataset(base_path='protGPT2_work-main/Bio_Dataset'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "nferruz/ProtGPT2"
    
    # Load model and tokenizer
    print(f"Loading model {model_name} on {device}...")
    protgpt2 = pipeline('text-generation', model=model_name, device_map="auto")
    tokenizer = protgpt2.tokenizer
    model = protgpt2.model.to(device)
    
    # Iterate through toxin folders
    for toxin_folder in tqdm(glob.glob(os.path.join(base_path, '*')), desc="Processing toxins"):
        if not os.path.isdir(toxin_folder):
            continue
        
        toxin_name = os.path.basename(toxin_folder)
        
        # Process each .txt file in the folder
        for txt_file in tqdm(glob.glob(os.path.join(toxin_folder, '*.txt')), desc=f"Processing {toxin_name}"):
            # Read existing sequences
            existing_seqs = preprocess_fasta(txt_file)
            current_count = len(existing_seqs)
            needed = max(150 - current_count, 0)
            
            if needed == 0:
                continue
            
            # Generate new sequences
            new_seqs = generate_sequences(model, tokenizer, needed, device)
            
            # Append to original file with headers
            with open(txt_file, 'a') as f:
                for i, seq in enumerate(new_seqs, 1):
                    header = f">Generated_{toxin_name}_sequence_{current_count + i}"
                    f.write(f"{header}\n{seq}\n")

if __name__ == "__main__":
    process_dataset()
