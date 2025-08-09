import argparse
import json
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--selection_metric", type=str, default="rougeL")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

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