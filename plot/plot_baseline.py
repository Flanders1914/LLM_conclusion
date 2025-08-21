#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DEFAULT_EVAL_ROOT = "output/eval_results"
DEFAULT_MODEL = "llama3.2-3b-base"
PROMPTS = [
    'Prompt0',
    'Prompt1',
    'Prompt2',
    'Prompt3'
]
METRICS = ['Rouge-L','Rouge-Lsum', 'Rouge-1', 'Rouge-2', 'Bleu', 'Meteor',
           'Word Pred Avg', 'Word Ref Avg']

def load_results(eval_root: str, model_name: str, prefix: str):
    """
    Read non_rct or rct results, return a list of dict:
    [{'Prompt': ..., 'Type': prefix, **metrics}, ...]
    """
    data = []
    for i, prompt in enumerate(PROMPTS):
        # Align with evaluate.sh outputs: output/eval_results/{MODEL}/prompt{i}/{prefix}_results.jsonl
        fn = os.path.join(eval_root, model_name, f"prompt{i}", f"{prefix}_results.jsonl")
        if not os.path.exists(fn):
            print(f"[WARNING] Missing evaluation file: {fn}. Skipping.")
            continue
        with open(fn, 'r', encoding='utf-8') as f:
            # First line contains overall metrics
            first_line = f.readline()
            if not first_line:
                print(f"[WARNING] Empty evaluation file: {fn}. Skipping.")
                continue
            res = json.loads(first_line)
        entry = {'Prompt': prompt, 'Type': prefix}
        entry.update({
            'Rouge-L':     res['rouge']['rougeL'],
            'Rouge-Lsum':  res['rouge']['rougeLsum'],
            'Rouge-1':     res['rouge']['rouge1'],
            'Rouge-2':     res['rouge']['rouge2'],
            'Bleu':        res['bleu']['bleu'],
            'Meteor':      res['meteor']['meteor'],
            'Word Pred Avg': res['word_count_predictions_avg'],
            'Word Ref Avg':  res['word_count_references_avg'],
        })
        data.append(entry)
    return data

def create_comprehensive_table(df, ax, model_name: str):
    """Create a comprehensive table showing all metrics for all prompts and dataset types

    The figure will also include the model name in the title for clarity.
    """
    # Create a multi-level table with all metrics
    # First, let's prepare the data structure
    prompts = df['Prompt'].unique()
    types = df['Type'].unique()
    
    # Create header rows
    header_row1 = ['Prompt', 'Dataset Type'] + METRICS
    
    # Prepare table data
    table_data = []
    for prompt in prompts:
        for dataset_type in types:
            row_data = df[(df['Prompt'] == prompt) & (df['Type'] == dataset_type)]
            if not row_data.empty:
                row = [prompt, dataset_type]
                for metric in METRICS:
                    value = row_data[metric].iloc[0]
                    row.append(f'{value:.4f}')
                table_data.append(row)
    
    # Combine headers and data
    all_data = [header_row1] + table_data
    
    # Create table
    table = ax.table(
        cellText=all_data,
        cellLoc='center',
        loc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Color header rows
    for i in range(len(header_row1)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color first two columns (row headers)
    for i in range(len(table_data) + 1):
        if i < 1:  # Skip header rows for first column
            continue
        table[(i, 0)].set_facecolor('#2196F3')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_facecolor('#2196F3')
        table[(i, 1)].set_text_props(weight='bold', color='white')
    
    # Remove axis
    ax.axis('off')
    ax.set_title(f'Comprehensive Results Table - All Metrics ({model_name})', fontsize=14, fontweight='bold', pad=20)
    # Add model info as a subtle figure-level title
    fig = ax.get_figure()
    if fig is not None:
        fig.suptitle(f"Model: {model_name}", fontsize=10, y=0.98, color='dimgray')

def main():
    parser = argparse.ArgumentParser(description="Plot baseline metrics across prompts from evaluation results")
    parser.add_argument("--eval_root", type=str, default=DEFAULT_EVAL_ROOT,
                        help="Root directory containing evaluation results (default: output/eval_results)")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="Model name subdirectory under eval_root (default: llama3.2-3b-base)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to save plots (default: eval_root/model_name)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.eval_root, args.model_name)
    os.makedirs(out_dir, exist_ok=True)

    # load two datasets
    non_rct = load_results(args.eval_root, args.model_name, 'non_rct')
    rct     = load_results(args.eval_root, args.model_name, 'rct')

    # merge
    df = pd.DataFrame(non_rct + rct)
    if df.empty:
        print("[ERROR] No evaluation data found. Ensure you have run evaluate.sh first.")
        return

    # plot each metric
    sns.set_style("whitegrid")
    for metric in METRICS:
        # Create bar plot figure
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()

        # Plot Non-RCT and RCT separately
        non_rct_data = df[df['Type'] == 'non_rct']
        rct_data = df[df['Type'] == 'rct']

        x = range(len(PROMPTS))
        width = 0.35

        bars1 = ax1.bar([i - width/2 for i in x], non_rct_data[metric], width,
                        label='Non-RCT', color='skyblue', alpha=0.8)
        bars2 = ax1.bar([i + width/2 for i in x], rct_data[metric], width,
                        label='RCT', color='lightcoral', alpha=0.8)

        ax1.set_title(f"{metric} Across Prompts", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Prompt Type", fontsize=12)
        ax1.set_ylabel(metric, fontsize=12)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(PROMPTS, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add model info at the figure level
        fig = plt.gcf()
        fig.suptitle(f"Model: {args.model_name}", fontsize=10, y=0.98, color='dimgray')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)

        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)

        # save bar plot file
        out_png = os.path.join(out_dir, f"{metric.replace(' ', '_')}.png")
        # leave room for the figure-level suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {out_png}")

    # Create comprehensive table with all metrics
    plt.figure(figsize=(16, 10))
    ax_table = plt.gca()
    create_comprehensive_table(df, ax_table, args.model_name)

    # save comprehensive table file
    table_png = os.path.join(out_dir, "comprehensive_results_table.png")
    # leave room for the figure-level suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(table_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {table_png}")


if __name__ == "__main__":
    main()