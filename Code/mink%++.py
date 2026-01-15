# -*- coding: utf-8 -*-
"""
Min-K%++ Memorization Detection Implementation
Modified to accept command-line arguments
"""

import logging
logging.basicConfig(level='ERROR')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

class MinKPlusPlusDetector:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate device mapping
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                return_dict=True,
                device_map='auto',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                return_dict=True,
                trust_remote_code=True
            )
            self.model.to(self.device)

        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
        print(f"Vocabulary size: {len(self.tokenizer)}")

    def calculate_mink_plus_plus_scores(self, text):
        # Tokenize the input text
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        input_ids = input_ids.to(self.device)

        mink_plus_plus_scores = []
        token_info = []

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]

            for i in range(len(input_ids[0]) - 1):
                current_logits = logits[i]
                target_token_id = input_ids[0][i + 1].item()

                log_probs = F.log_softmax(current_logits, dim=-1)
                probs = F.softmax(current_logits, dim=-1)
                mu = torch.sum(probs * log_probs).item()

                variance = torch.sum(probs * (log_probs - mu) ** 2).item()
                sigma = np.sqrt(variance)

                target_log_prob = log_probs[target_token_id].item()

                if sigma > 0:
                    mink_plus_plus_score = (target_log_prob - mu) / sigma
                else:
                    mink_plus_plus_score = 0.0

                mink_plus_plus_scores.append(mink_plus_plus_score)

                token_info.append({
                    'position': i + 1,
                    'token': self.tokenizer.decode([target_token_id]),
                    'token_id': target_token_id,
                    'log_prob': target_log_prob,
                    'mu': mu,
                    'sigma': sigma,
                    'mink_plus_plus_score': mink_plus_plus_score
                })

        return mink_plus_plus_scores, token_info

    def min_k_plus_plus(self, text, k_percent=20):
        text = text.replace('\x00', '').strip()
        if not text:
            return None

        mink_plus_plus_scores, token_info = self.calculate_mink_plus_plus_scores(text)

        if not mink_plus_plus_scores:
            return None

        k_length = int(len(mink_plus_plus_scores) * (k_percent / 100))
        if k_length == 0:
            k_length = 1

        sorted_scores = sorted(mink_plus_plus_scores)
        min_k_scores = sorted_scores[:k_length]
        min_k_plus_plus_score = np.mean(min_k_scores)

        metrics = {
            'text': text[:200] + '...' if len(text) > 200 else text,
            'text_length': len(text),
            'num_tokens': len(mink_plus_plus_scores),
            'k_percent': k_percent,
            'k_length': k_length,
            f'min_{k_percent}_plus_plus': min_k_plus_plus_score,
            'mean_score': np.mean(mink_plus_plus_scores),
            'std_score': np.std(mink_plus_plus_scores),
            'min_score': np.min(mink_plus_plus_scores),
            'max_score': np.max(mink_plus_plus_scores),
            'median_score': np.median(mink_plus_plus_scores),
            'q25_score': np.percentile(mink_plus_plus_scores, 25),
            'q75_score': np.percentile(mink_plus_plus_scores, 75),
        }

        for k in [5, 10, 20, 30, 40, 50]:
            k_len = int(len(mink_plus_plus_scores) * (k / 100))
            if k_len == 0:
                k_len = 1
            min_k_scores_temp = sorted(mink_plus_plus_scores)[:k_len]
            metrics[f'min_{k}_plus_plus'] = np.mean(min_k_scores_temp)

        return metrics

    def analyze_dataset(self, texts, k_percent=20, max_texts=None):
        if max_texts:
            texts = texts[:max_texts]

        results = []
        print(f"Analyzing {len(texts)} texts with Min-{k_percent}%++...")

        for i, text in enumerate(tqdm(texts, desc="Processing texts")):
            try:
                metrics = self.min_k_plus_plus(text, k_percent)
                if metrics:
                    metrics['text_id'] = i
                    results.append(metrics)
                else:
                    print(f"Skipped text {i}: empty or invalid")

            except Exception as e:
                print(f"Error processing text {i}: {str(e)}")
                continue

            if (i + 1) % 20 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return results

    def create_summary_statistics(self, results):
        if not results:
            return {}

        summary = {
            'total_texts': len(results),
        }

        for k in [5, 10, 20, 30, 40, 50]:
            k_scores = [r[f'min_{k}_plus_plus'] for r in results if f'min_{k}_plus_plus' in r]
            if k_scores:
                summary[f'min_{k}_plus_plus_stats'] = {
                    'mean': np.mean(k_scores),
                    'std': np.std(k_scores),
                    'min': np.min(k_scores),
                    'max': np.max(k_scores),
                    'median': np.median(k_scores),
                    'q25': np.percentile(k_scores, 25),
                    'q75': np.percentile(k_scores, 75)
                }
            else:
                summary[f'min_{k}_plus_plus_stats'] = {}

        return summary

def load_legal_dataset(file_path):
    try:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded dataset with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            print("Failed to load dataset with common encodings")
            return []

        if 'Input' not in df.columns:
            print(f"Available columns: {df.columns.tolist()}")
            print("Using the first column as input text")
            texts = df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            texts = df['Input'].dropna().astype(str).tolist()

        cleaned_texts = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                cleaned_texts.append(text.strip())

        print(f"Loaded {len(cleaned_texts)} texts from {file_path}")
        return cleaned_texts

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return []

def plot_results(results, save_path='mink_plus_plus_results.png'):
    if not results:
        print("No results to plot")
        return

    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    k_values = [5, 10, 20, 30, 40, 50]
    k_scores = {}
    for k in k_values:
        k_scores[k] = [r[f'min_{k}_plus_plus'] for r in results if f'min_{k}_plus_plus' in r]

    # Plot 1: Boxplot
    bp = axes[0, 0].boxplot([k_scores[k] for k in k_values], labels=[f'{k}%' for k in k_values])
    axes[0, 0].set_title('Min-K%++ Score Distribution by K Value', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('K Percentage')
    axes[0, 0].set_ylabel('Min-K%++ Score')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Histogram
    min_20_scores = k_scores[20]
    axes[0, 1].hist(min_20_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Min-20%++ Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Min-20%++ Score')
    axes[0, 1].set_ylabel('Frequency')
    mean_score = np.mean(min_20_scores)
    axes[0, 1].axvline(mean_score, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_score:.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Trend line
    mean_scores = [np.mean(k_scores[k]) for k in k_values]
    axes[0, 2].plot(k_values, mean_scores, 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 2].set_title('Mean Min-K%++ Score vs K Value', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('K Percentage')
    axes[0, 2].set_ylabel('Mean Min-K%++ Score')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Scatter plot
    text_lengths = [r['num_tokens'] for r in results]
    axes[1, 0].scatter(text_lengths, min_20_scores, alpha=0.6, color='orange')
    axes[1, 0].set_title('Text Length vs Min-20%++ Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Tokens')
    axes[1, 0].set_ylabel('Min-20%++ Score')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Violin plot
    selected_k = [10, 20, 30]
    data_for_violin = [k_scores[k] for k in selected_k]
    parts = axes[1, 1].violinplot(data_for_violin, positions=selected_k, widths=5, showmeans=True)
    axes[1, 1].set_title('Score Distribution Comparison (Violin Plot)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('K Percentage')
    axes[1, 1].set_ylabel('Min-K%++ Score')
    axes[1, 1].set_xticks(selected_k)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Pie chart
    threshold_low = np.percentile(min_20_scores, 25)
    threshold_high = np.percentile(min_20_scores, 75)

    categories = []
    for score in min_20_scores:
        if score <= threshold_low:
            categories.append('High Memorization\nLikelihood')
        elif score >= threshold_high:
            categories.append('Low Memorization\nLikelihood')
        else:
            categories.append('Medium Memorization\nLikelihood')

    from collections import Counter
    cat_counts = Counter(categories)
    axes[1, 2].pie(cat_counts.values(), labels=cat_counts.keys(), autopct='%1.1f%%',
                  colors=['#ff9999', '#ffcc99', '#99ff99'])
    axes[1, 2].set_title('Memorization Likelihood Classification\n(Based on Min-20%++ Scores)',
                        fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

def save_results_to_json(results, summary, model_name, file_path='mink_plus_plus_results.json'):
    output_data = {
        'summary': summary,
        'detailed_results': results,
        'metadata': {
            'total_texts_analyzed': len(results),
            'model_used': model_name,
            'method': 'Min-K%++',
            'k_percentage': 20,
        }
    }

    with open(file_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Results saved to {file_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Min-K++ Memorization Detection Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python mink_plus_plus.py --file my_data.csv --model "meta-llama/Llama-2-7b-hf"
  python mink_plus_plus.py --file my_data.csv --max-texts 50 --k-percent 30
        '''
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        required=True,
        help='Path to the CSV dataset file'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path or name of the HuggingFace model'
    )
    
    parser.add_argument(
        '--max-texts', '-n',
        type=int,
        default=None,
        help='Maximum number of texts to analyze (default: all)'
    )
    
    parser.add_argument(
        '--k-percent', '-k',
        type=int,
        default=20,
        help='K percentage value for Min-K++ calculation (default: 20)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='mink_plus_plus_results',
        help='Output file prefix for results (default: mink_plus_plus_results)'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the Min-K++ analysis"""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    print("=" * 80)
    print("Min-K%++ Memorization Detection")
    print("PRE-TRAINING DATA FROM LARGE LANGUAGE MODELS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Dataset file: {args.file}")
    print(f"  Model: {args.model}")
    print(f"  Max texts: {args.max_texts if args.max_texts else 'All'}")
    print(f"  K percentage: {args.k_percent}")
    print(f"  Output prefix: {args.output}")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading dataset...")
    texts = load_legal_dataset(args.file)

    if not texts:
        print(f"Failed to load dataset from {args.file}")
        sys.exit(1)

    print(f"Dataset loaded: {len(texts)} texts")
    print(f"Sample text preview: {texts[0][:200]}..." if texts else "No texts found")
    print(f"Average text length: {np.mean([len(text.split()) for text in texts]):.1f} words")

    # Initialize detector
    print("\n2. Initializing Min-K%++ detector...")
    try:
        detector = MinKPlusPlusDetector(model_name=args.model)
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        sys.exit(1)

    # Run analysis
    print(f"\n3. Running Min-K%++ analysis (k={args.k_percent})...")
    results = detector.analyze_dataset(texts, k_percent=args.k_percent, max_texts=args.max_texts)

    if not results:
        print("No results generated. Check for errors in processing.")
        sys.exit(1)

    print(f"Successfully analyzed {len(results)} texts")

    # Generate summary statistics
    print("\n4. Generating summary statistics...")
    summary = detector.create_summary_statistics(results)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY - Min-K%++ Method")
    print("=" * 80)
    print(f"Total texts analyzed: {summary['total_texts']}")

    print(f"\nMin-{args.k_percent}%++ Score Statistics:")
    stats = summary[f'min_{args.k_percent}_plus_plus_stats']
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")
    print(f"  Min:  {stats['min']:.4f}")
    print(f"  Max:  {stats['max']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Q25: {stats['q25']:.4f}")
    print(f"  Q75: {stats['q75']:.4f}")

    print(f"\nComparison across different K values:")
    for k in [5, 10, 20, 30, 40, 50]:
        k_stats = summary[f'min_{k}_plus_plus_stats']
        print(f"  Min-{k:2d}%++: Mean = {k_stats['mean']:7.4f}, Std = {k_stats['std']:7.4f}")

    # Show sample results
    print(f"\nSample Results (first 5 texts):")
    print("-" * 80)
    for i, result in enumerate(results[:5]):
        print(f"Text {i+1}:")
        print(f"  Preview: {result['text'][:100]}...")
        print(f"  Min-{args.k_percent}%++ Score: {result[f'min_{args.k_percent}_plus_plus']:8.4f}")
        print(f"  Tokens: {result['num_tokens']:4d}")
        print()

    # Generate visualizations
    print("5. Generating visualizations...")
    plot_results(results, save_path=f"{args.output}.png")

    # Save results
    print("6. Saving results...")
    save_results_to_json(results, summary, args.model, file_path=f"{args.output}.json")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("Files generated:")
    print(f"- {args.output}.png (visualizations)")
    print(f"- {args.output}.json (detailed results)")

if __name__ == "__main__":
    main()
