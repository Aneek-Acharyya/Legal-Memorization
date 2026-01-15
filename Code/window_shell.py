# -*- coding: utf-8 -*-
"""
Training Data Extraction - Memorization Detection
Based on Carlini et al. (2021)
"""

import torch
import numpy as np
import pandas as pd
import zlib
import logging
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.ERROR)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Memorization Detection in Language Models')
    
    parser.add_argument('--model_name', type=str, required=True,
                        help='Path or name of the model to load')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the input CSV dataset')
    parser.add_argument('--output_prefix', type=str, default='memorization_results',
                        help='Prefix for output files (default: memorization_results)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of top samples to analyze (default: 100)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--use_quantization', action='store_true',
                        help='Use 4-bit quantization for memory efficiency')
    
    return parser.parse_args()

def calculate_perplexity(text, model, tokenizer, max_length=512):
    """Calculate perplexity of a text sequence using the model."""
    try:
        encodings = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=False
        )
        
        input_ids = encodings['input_ids'].to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        return perplexity
    
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return float('inf')

def calculate_zlib_entropy(text):
    """Calculate zlib compression entropy of text."""
    try:
        compressed = zlib.compress(bytes(text, 'utf-8'))
        return len(compressed)
    except Exception as e:
        print(f"Error calculating zlib entropy: {e}")
        return float('inf')

def compute_all_metrics(text, model, tokenizer, max_length=512):
    """Compute all membership inference metrics for a given text."""
    metrics = {}
    
    # 1. Calculate perplexity
    ppl = calculate_perplexity(text, model, tokenizer, max_length)
    metrics['perplexity'] = ppl
    metrics['log_perplexity'] = np.log(ppl) if ppl > 0 and ppl != float('inf') else float('inf')
    
    # 2. Calculate zlib entropy
    zlib_entropy = calculate_zlib_entropy(text)
    metrics['zlib_entropy'] = zlib_entropy
    
    # 3. Calculate zlib ratio
    if metrics['log_perplexity'] != float('inf') and metrics['log_perplexity'] > 0:
        metrics['zlib_ratio'] = zlib_entropy / metrics['log_perplexity']
    else:
        metrics['zlib_ratio'] = 0
    
    # 4. Calculate lowercase perplexity
    ppl_lower = calculate_perplexity(text.lower(), model, tokenizer, max_length)
    metrics['perplexity_lower'] = ppl_lower
    metrics['log_perplexity_lower'] = np.log(ppl_lower) if ppl_lower > 0 and ppl_lower != float('inf') else float('inf')
    
    # 5. Calculate lowercase ratio
    if metrics['log_perplexity'] != float('inf') and metrics['log_perplexity'] > 0:
        metrics['lowercase_ratio'] = metrics['log_perplexity_lower'] / metrics['log_perplexity']
    else:
        metrics['lowercase_ratio'] = 1.0
    
    return metrics

def extract_memorization_scores(dataset, model, tokenizer, num_samples=None, max_length=512):
    """Extract memorization scores for all samples in the dataset."""
    print("\nComputing memorization metrics for dataset samples...")
    
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.head(num_samples).copy()
    else:
        dataset = dataset.copy()
    
    # Initialize metric columns
    dataset['perplexity'] = 0.0
    dataset['log_perplexity'] = 0.0
    dataset['zlib_entropy'] = 0
    dataset['zlib_ratio'] = 0.0
    dataset['perplexity_lower'] = 0.0
    dataset['log_perplexity_lower'] = 0.0
    dataset['lowercase_ratio'] = 0.0
    
    # Process each sample
    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        text = dataset.iloc[idx]['Input']
        
        metrics = compute_all_metrics(text, model, tokenizer, max_length)
        
        dataset.at[dataset.index[idx], 'perplexity'] = metrics['perplexity']
        dataset.at[dataset.index[idx], 'log_perplexity'] = metrics['log_perplexity']
        dataset.at[dataset.index[idx], 'zlib_entropy'] = metrics['zlib_entropy']
        dataset.at[dataset.index[idx], 'zlib_ratio'] = metrics['zlib_ratio']
        dataset.at[dataset.index[idx], 'perplexity_lower'] = metrics['perplexity_lower']
        dataset.at[dataset.index[idx], 'log_perplexity_lower'] = metrics['log_perplexity_lower']
        dataset.at[dataset.index[idx], 'lowercase_ratio'] = metrics['lowercase_ratio']
    
    return dataset

def analyze_and_rank_samples(results_df, top_n=100):
    """Analyze and rank samples based on different memorization metrics."""
    print(f"\n{'='*80}")
    print("MEMORIZATION DETECTION ANALYSIS")
    print(f"{'='*80}")
    
    rankings = {}
    
    # 1. Rank by lowest perplexity
    print(f"\n{'='*80}")
    print("1. TOP SAMPLES BY LOWEST PERPLEXITY")
    print(f"{'='*80}")
    
    df_ppl = results_df.nsmallest(top_n, 'perplexity').copy()
    rankings['perplexity'] = df_ppl
    
    print(f"\nTop 10 samples with lowest perplexity:")
    for i, row in df_ppl.head(10).iterrows():
        print(f"\nRank {list(df_ppl.index).index(i) + 1}:")
        print(f"  Perplexity: {row['perplexity']:.4f}")
        print(f"  Text: {row['Input'][:100]}...")
    
    # 2. Rank by highest zlib ratio
    print(f"\n{'='*80}")
    print("2. TOP SAMPLES BY HIGHEST ZLIB RATIO")
    print(f"{'='*80}")
    
    df_zlib = results_df.nlargest(top_n, 'zlib_ratio').copy()
    rankings['zlib_ratio'] = df_zlib
    
    print(f"\nTop 10 samples with highest zlib ratio:")
    for i, row in df_zlib.head(10).iterrows():
        print(f"\nRank {list(df_zlib.index).index(i) + 1}:")
        print(f"  Zlib Ratio: {row['zlib_ratio']:.4f}")
        print(f"  Text: {row['Input'][:100]}...")
    
    # 3. Rank by lowercase ratio
    print(f"\n{'='*80}")
    print("3. TOP SAMPLES BY LOWERCASE RATIO")
    print(f"{'='*80}")
    
    df_lower = results_df.nlargest(top_n, 'lowercase_ratio').copy()
    rankings['lowercase_ratio'] = df_lower
    
    print(f"\nTop 10 samples with highest lowercase ratio:")
    for i, row in df_lower.head(10).iterrows():
        print(f"\nRank {list(df_lower.index).index(i) + 1}:")
        print(f"  Lowercase Ratio: {row['lowercase_ratio']:.4f}")
        print(f"  Text: {row['Input'][:100]}...")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nPerplexity Statistics:")
    print(results_df['perplexity'].describe())
    
    print(f"\nZlib Ratio Statistics:")
    print(results_df['zlib_ratio'].describe())
    
    print(f"\nLowercase Ratio Statistics:")
    print(results_df['lowercase_ratio'].describe())
    
    return rankings

def save_results(results_df, rankings, output_prefix='memorization_results'):
    """Save results to JSON files."""
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    complete_output = f"{output_prefix}_complete.json"
    results_df.to_json(complete_output, orient='records', indent=4)
    print(f"✓ Complete results saved to: {complete_output}")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    print(f"\n{'='*80}")
    print("TRAINING DATA EXTRACTION - MEMORIZATION DETECTION")
    print("Based on Carlini et al. (2021)")
    print(f"{'='*80}")
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load dataset
    print(f"\nLoading dataset from: {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    print(f"Dataset loaded with {len(df)} samples")
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    if args.use_quantization:
        print("Using 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    
    model.eval()
    print("Model loaded successfully!")
    
    print(f"\nConfiguration:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset samples: {len(df) if args.num_samples is None else args.num_samples}")
    print(f"  - Top N for ranking: {args.top_n}")
    print(f"  - Max sequence length: {args.max_length}")
    print(f"  - Output prefix: {args.output_prefix}")
    
    # Run extraction
    results = extract_memorization_scores(
        dataset=df,
        model=model,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        max_length=args.max_length
    )
    
    # Analyze and rank
    rankings = analyze_and_rank_samples(results, top_n=args.top_n)
    
    # Save results
    save_results(results, rankings, output_prefix=args.output_prefix)
    
    print(f"\n{'='*80}")
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    
    return results, rankings

if __name__ == "__main__":
    main()
