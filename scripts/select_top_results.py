import argparse
import json
import sys

def compare_results_and_select_top(file1_path, file2_path, output_path, selection_metric="rougeL", top_n=10):
    """
    Compare two result files and select top results based on the difference of the metric.
    
    Args:
        file1_path (str): Path to the first result file
        file2_path (str): Path to the second result file
        output_path (str): Path to save the comparison results
        selection_metric (str): Metric to use for comparison
        top_n (int): Number of top results to select
    
    Returns:
        dict: Dictionary containing top results with largest differences
    """
    # Read first file
    results1 = []
    with open(file1_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if line_num == 1:
                continue  # skip average scores
            data = json.loads(line)
            results1.append(data)
    
    # Read second file
    results2 = []
    with open(file2_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if line_num == 1:
                continue  # skip average scores
            data = json.loads(line)
            results2.append(data)
    
    # Validate files have same number of entries
    if len(results1) != len(results2):
        raise ValueError(f"Files have different number of entries: {len(results1)} vs {len(results2)}")
    
    # Validate metric
    if selection_metric not in ["rougeL", "rougeLsum", "rouge1", "rouge2", "meteor", "word_count_prediction"]:
        raise ValueError(f"Invalid selection metric: {selection_metric}")
    
    # Calculate differences
    differences = []
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        # Extract metric values
        if selection_metric in ["rougeL", "rouge1", "rouge2", "rougeLsum"]:
            value1 = r1["rouge"][selection_metric]
            value2 = r2["rouge"][selection_metric]
        elif selection_metric == "meteor":
            value1 = r1["meteor"][selection_metric]
            value2 = r2["meteor"][selection_metric]
        elif selection_metric == "word_count_prediction":
            value1 = r1["word_count_prediction"]
            value2 = r2["word_count_prediction"]
        
        difference = value2 - value1  # positive means file2 is better
        
        # Store comparison data
        comparison_entry = {
            "index": i,
            "line_number": r1["line_number"],
            "input": r1["input"],
            "prediction1": r1["prediction"],
            "prediction2": r2["prediction"],
            "reference": r1["reference"],
            "metric_value1": value1,
            "metric_value2": value2,
            "difference": difference,
            "rouge1": r1["rouge"],
            "rouge2": r2["rouge"],
            "meteor1": r1["meteor"],
            "meteor2": r2["meteor"],
            "word_count_prediction1": r1["word_count_prediction"],
            "word_count_prediction2": r2["word_count_prediction"],
            "word_count_reference": r1["word_count_reference"]
        }
        differences.append(comparison_entry)
    
    # Sort by difference (descending - largest improvements first)
    differences.sort(key=lambda x: x["difference"], reverse=True)
    
    # Select top n with largest positive differences (biggest improvements)
    top_improvements = differences[:top_n]
    
    # Select top n with largest negative differences (biggest degradations)
    top_degradations = differences[-top_n:]
    top_degradations.reverse()  # Make it largest degradation first
    
    # Print results
    print("=" * 80)
    print(f"COMPARISON RESULTS: {selection_metric} differences (File2 - File1)")
    print("=" * 80)
    
    print(f"\nTOP {top_n} LARGEST IMPROVEMENTS (File2 > File1):")
    print("-" * 60)
    for i, entry in enumerate(top_improvements, 1):
        print(f"\nRank {i}:")
        print(f"  Line Number: {entry['line_number']}")
        print(f"  Input:\n{entry['input']}\n")
        print(f"  Prediction (File1):\n{entry['prediction1']}\n")
        print(f"  Prediction (File2):\n{entry['prediction2']}\n")
        print(f"  Reference:\n{entry['reference']}\n")
        print(f"  {selection_metric} (File1): {entry['metric_value1']:.4f}")
        print(f"  {selection_metric} (File2): {entry['metric_value2']:.4f}")
        print(f"  Difference: +{entry['difference']:.4f}")
        print("-" * 60)
    
    print(f"\nTOP {top_n} LARGEST DEGRADATIONS (File1 > File2):")
    print("-" * 60)
    for i, entry in enumerate(top_degradations, 1):
        print(f"\nRank {i}:")
        print(f"  Line Number: {entry['line_number']}")
        print(f"  Input:\n{entry['input']}\n")
        print(f"  Prediction (File1):\n{entry['prediction1']}\n")
        print(f"  Prediction (File2):\n{entry['prediction2']}\n")
        print(f"  Reference:\n{entry['reference']}\n")
        print(f"  {selection_metric} (File1): {entry['metric_value1']:.4f}")
        print(f"  {selection_metric} (File2): {entry['metric_value2']:.4f}")
        print(f"  Difference: {entry['difference']:.4f}")
        print("-" * 60)
    
    # Save results
    result = {
        "comparison_metric": selection_metric,
        "file1_path": file1_path,
        "file2_path": file2_path,
        "total_entries": len(differences),
        "top_improvements": top_improvements,
        "top_degradations": top_degradations,
        "all_differences": differences  # Include all for further analysis
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nComparison results saved to: {output_path}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select top results from evaluation files or compare two result files")
    parser.add_argument("--input_path", type=str, help="Path to input file (for single file analysis)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output")
    parser.add_argument("--selection_metric", type=str, default="rougeL", 
                       help="Metric to use for selection/comparison")
    parser.add_argument("--top_n", type=int, default=10, 
                       help="Number of top results to select")
    
    # New arguments for comparison mode
    parser.add_argument("--compare", action="store_true", 
                       help="Enable comparison mode between two files")
    parser.add_argument("--file1_path", type=str, 
                       help="Path to first file for comparison")
    parser.add_argument("--file2_path", type=str, 
                       help="Path to second file for comparison")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.compare:
        if not args.file1_path or not args.file2_path:
            print("Error: --file1_path and --file2_path are required when using --compare mode")
            sys.exit(1)
        
        # Run comparison mode
        try:
            compare_results_and_select_top(
                args.file1_path, 
                args.file2_path, 
                args.output_path, 
                args.selection_metric, 
                args.top_n
            )
        except Exception as e:
            print(f"Error during comparison: {e}")
            sys.exit(1)
        
        sys.exit(0)  # Exit after comparison
    
    # Original single file analysis mode
    if not args.input_path:
        print("Error: --input_path is required for single file analysis mode")
        sys.exit(1)

    individual_results = []
    n = args.top_n

    # read the input file
    with open(args.input_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if line_num == 1:
                # skip the first line(average scores)
                continue
            data = json.loads(line)
            individual_results.append(data)


    if 2 * n > len(individual_results):
        print(f"The number of data is less than 2 * n!")
        sys.exit(1)

    if args.selection_metric not in ["rougeL","rougeLsum", "rouge1", "rouge2", "meteor", "word_count_prediction"]:
        print(f"Invalid selection metric: {args.selection_metric}")
        sys.exit(1)
    
    worst_n_entries = []
    best_n_entries = []
    
    # Sort by selection metric (ascending - worst first)
    if args.selection_metric == "rougeL" or args.selection_metric == "rouge1" or args.selection_metric == "rouge2" or args.selection_metric == "rougeLsum":
        individual_results.sort(key=lambda x: x["rouge"][args.selection_metric])
    elif args.selection_metric == "meteor":
        individual_results.sort(key=lambda x: x["meteor"][args.selection_metric])
    elif args.selection_metric == "word_count_prediction":
        individual_results.sort(key=lambda x: x["word_count_prediction"])
    
    # Output top n worst performers (smallest ROUGE-L scores)
    print("--------------------------------")
    print(f"TOP {n} ENTRIES WITH SMALLEST {args.selection_metric} SCORES:")
    print("--------------------------------")
    for i, result in enumerate(individual_results[:n], 1):
        worst_n_entries.append(result)
        print(f"Rank {i}:")
        print(f"  Line Number: {result['line_number']}")
        print(f"  Input:\n{result['input']}\n")
        print(f"  Prediction:\n{result['prediction']}\n")
        print(f"  Reference:\n{result['reference']}\n")    
        print(f"  ROUGE-L Score: {result['rouge']['rougeL']:.4f}")
        print(f"  ROUGE-L-Sum Score: {result['rouge']['rougeLsum']:.4f}")
        print(f"  ROUGE-1 Score: {result['rouge']['rouge1']:.4f}")
        print(f"  ROUGE-2 Score: {result['rouge']['rouge2']:.4f}")
        print(f"  METEOR Score: {result['meteor']['meteor']:.4f}")
        print(f"  Word Count Prediction: {result['word_count_prediction']}")
        print(f"  Word Count Reference: {result['word_count_reference']}")
        print("--------------------------------")
    
    # Output top n best performers (largest ROUGE-L scores)
    print("--------------------------------")
    print(f"TOP {n} ENTRIES WITH LARGEST {args.selection_metric} SCORES:")
    print("--------------------------------")
    for i, result in enumerate(reversed(individual_results[-n:]), 1):
        best_n_entries.append(result)
        print(f"Rank {i}:")
        print(f"  Line Number: {result['line_number']}")
        print(f"  Input:\n{result['input']}\n")
        print(f"  Prediction:\n{result['prediction']}\n")
        print(f"  Reference:\n{result['reference']}\n")    
        print(f"  ROUGE-L Score: {result['rouge']['rougeL']:.4f}")
        print(f"  ROUGE-L-Sum Score: {result['rouge']['rougeLsum']:.4f}")
        print(f"  ROUGE-1 Score: {result['rouge']['rouge1']:.4f}")
        print(f"  ROUGE-2 Score: {result['rouge']['rouge2']:.4f}")
        print(f"  METEOR Score: {result['meteor']['meteor']:.4f}")
        print(f"  Word Count Prediction: {result['word_count_prediction']}")
        print(f"  Word Count Reference: {result['word_count_reference']}")
        print("--------------------------------")

    # save the results
    result = {
        "top_n_worst_entries": worst_n_entries,
        "top_n_best_entries": best_n_entries
    }
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)