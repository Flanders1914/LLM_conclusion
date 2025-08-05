#!/usr/bin/env python3
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = "output/predictions/llama3_8b_base"
PROMPTS = [
    'Prompt0',
    'Prompt1 (+context)',
    'Prompt2 (+length restriction)',
    'Prompt3 (+context & length restriction)'
]
METRICS = ['Rouge-L', 'Rouge-1', 'Rouge-2', 'Bleu', 'Meteor',
           'Word Count Predictions Avg', 'Word Count References Avg']

def load_results(prefix: str):
    """
    read non_rct or rct results, return a list of dict:
    [{'Prompt': ..., 'Type': prefix, **metrics}, ...]
    """
    data = []
    for i, prompt in enumerate(PROMPTS):
        fn = os.path.join(BASE_DIR, f"{prefix}_prompt{i}_result.json")
        with open(fn, 'r') as f:
            res = json.load(f)
        entry = {'Prompt': prompt, 'Type': prefix}
        entry.update({
            'Rouge-L':     res['rouge']['rougeL'],
            'Rouge-1':     res['rouge']['rouge1'],
            'Rouge-2':     res['rouge']['rouge2'],
            'Bleu':        res['bleu']['bleu'],
            'Meteor':      res['meteor']['meteor'],
            'Word Count Predictions Avg': res['word_count_predictions_avg'],
            'Word Count References Avg':  res['word_count_references_avg'],
        })
        data.append(entry)
    return data

def create_comprehensive_table(df, ax):
    """Create a comprehensive table showing all metrics for all prompts and dataset types"""
    # Create a multi-level table with all metrics
    # First, let's prepare the data structure
    prompts = df['Prompt'].unique()
    types = df['Type'].unique()
    
    # Create header rows
    header_row1 = ['Prompt', 'Dataset Type'] + METRICS
    header_row2 = ['', ''] + [f'({metric})' for metric in METRICS]
    
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
    all_data = [header_row1, header_row2] + table_data
    
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
        table[(1, i)].set_facecolor('#4CAF50')
        table[(1, i)].set_text_props(weight='bold', color='white')
    
    # Color first two columns (row headers)
    for i in range(len(table_data) + 2):
        if i < 2:  # Skip header rows for first column
            continue
        table[(i, 0)].set_facecolor('#2196F3')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_facecolor('#2196F3')
        table[(i, 1)].set_text_props(weight='bold', color='white')
    
    # Remove axis
    ax.axis('off')
    ax.set_title('Comprehensive Results Table - All Metrics', fontsize=14, fontweight='bold', pad=20)

# load two datasets
non_rct = load_results('non_rct')
rct     = load_results('rct')

# merge
df = pd.DataFrame(non_rct + rct)

# plot each metric
sns.set_style("whitegrid")
for metric in METRICS:
    # Create bar plot figure
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    
    # Create bar plot without hue (Dataset type out of figure)
    # We'll plot Non-RCT and RCT separately
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
    ax1.set_xticks(x)
    ax1.set_xticklabels(PROMPTS, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
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
    out_png = os.path.join(BASE_DIR, f"{metric.replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {out_png}")

# Create comprehensive table with all metrics
plt.figure(figsize=(16, 10))
ax_table = plt.gca()
create_comprehensive_table(df, ax_table)

# save comprehensive table file
table_png = os.path.join(BASE_DIR, "comprehensive_results_table.png")
plt.tight_layout()
plt.savefig(table_png, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: {table_png}")