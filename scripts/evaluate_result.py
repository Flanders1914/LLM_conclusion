import evaluate
import argparse
import os
import json
import nltk
import sys
nltk.download("wordnet")
nltk.download("omw-1.4")

def evaluate_rouge(predictions, references, rouge):
    references = [[r] for r in references]
    results = rouge.compute(predictions=predictions, references=references)
    return results

def evaluate_rouge_individual(prediction, reference, rouge):
    """Evaluate ROUGE score for a single prediction-reference pair"""
    results = rouge.compute(predictions=[prediction], references=[[reference]])
    return results

def evaluate_bleu(predictions, references, bleu):
    references = [[r] for r in references]
    results = bleu.compute(predictions=predictions, references=references)
    return results

def evaluate_meteor(predictions, references, meteor):
    references = [[r] for r in references]
    results = meteor.compute(predictions=predictions, references=references)
    return results

def average_word_count(list_of_text: list[str]):
    return sum(len(text.split()) for text in list_of_text) / len(list_of_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    # create the output directory if it doesn't exist
    dirpath = os.path.dirname(args.output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    predictions = []
    references = []
    individual_results = []

    # read the input file
    with open(args.input_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            data = json.loads(line)
            # get the predictions and references
            prediction = data["output"]
            reference = data["answer"]
            input = data["input"]
            predictions.append(prediction)
            references.append(reference)
            
            # Store original data for individual analysis
            individual_results.append({
                "line_number": line_num,
                "input": input,
                "prediction": prediction,
                "reference": reference
            })

    # evaluate the results
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    print(f"Start evaluating the results in {args.input_path}")
    print("--------------------------------")
    print(f"The number of data is {len(predictions)}")
    print("--------------------------------")

    n = args.top_n
    if 2 * n > len(individual_results):
        print(f"The number of data is less than 2 * n!")
        sys.exit(1)
    
    # Evaluate individual ROUGE scores
    worst_n_entries = []
    best_n_entries = []
    print("Evaluating individual ROUGE scores...")
    for i, result in enumerate(individual_results):
        if i % 100 == 0:
            print(f"Evaluated {i} entries")
        rouge_scores = evaluate_rouge_individual(result["prediction"], result["reference"], rouge)
        individual_results[i]["rouge_scores"] = rouge_scores
    
    # Sort by ROUGE-L score (ascending - worst first)
    individual_results.sort(key=lambda x: x["rouge_scores"]["rougeL"])
    
    # Output top n worst performers (smallest ROUGE-L scores)
    print("--------------------------------")
    print("TOP n ENTRIES WITH SMALLEST ROUGE-L SCORES:")
    print("--------------------------------")
    for i, result in enumerate(individual_results[:n], 1):
        worst_n_entries.append(result)
        print(f"Rank {i}:")
        print(f"  Line Number: {result['line_number']}")
        print(f"  ROUGE-L Score: {result['rouge_scores']['rougeL']:.4f}")
        print(f"  ROUGE-1 Score: {result['rouge_scores']['rouge1']:.4f}")
        print(f"  ROUGE-2 Score: {result['rouge_scores']['rouge2']:.4f}")
        print(f"  Input: {result['input']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Reference: {result['reference']}")
        print("--------------------------------")
    
    # Output top n best performers (largest ROUGE-L scores)
    print("--------------------------------")
    print("TOP n ENTRIES WITH LARGEST ROUGE-L SCORES:")
    print("--------------------------------")
    for i, result in enumerate(reversed(individual_results[-n:]), 1):
        best_n_entries.append(result)
        print(f"Rank {i}:")
        print(f"  Line Number: {result['line_number']}")    
        print(f"  ROUGE-L Score: {result['rouge_scores']['rougeL']:.4f}")
        print(f"  ROUGE-1 Score: {result['rouge_scores']['rouge1']:.4f}")
        print(f"  ROUGE-2 Score: {result['rouge_scores']['rouge2']:.4f}")
        print(f"  Input: {result['input']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Reference: {result['reference']}")
        print("--------------------------------")
    
    print("Evaluate the overall results")
    print("--------------------------------")
    rouge_results = evaluate_rouge(predictions, references, rouge)
    print(f"The overall rouge results are: ROUGE-L: {rouge_results['rougeL']:.4f}, ROUGE-1: {rouge_results['rouge1']:.4f}, ROUGE-2: {rouge_results['rouge2']:.4f}")
    print("--------------------------------")
    bleu_results = evaluate_bleu(predictions, references, bleu)
    print(f"The overall bleu results are: BLEU: {bleu_results['bleu']:.4f}, \
        Unigram: {bleu_results['precisions'][0]:.4f}, Bigram: {bleu_results['precisions'][1]:.4f}, Trigram: {bleu_results['precisions'][2]:.4f}, \
        Pentagram: {bleu_results['precisions'][3]:.4f}, Brevity Penalty: {bleu_results['brevity_penalty']:.4f}")
    print("--------------------------------")
    meteor_results = evaluate_meteor(predictions, references, meteor)
    print(f"The overall meteor results are: {meteor_results['meteor']:.4f}")
    print("--------------------------------")
    word_count_predictions_avg = average_word_count(predictions)
    print(f"The average word count of predictions is {word_count_predictions_avg}")
    word_count_references_avg = average_word_count(references)
    print(f"The average word count of references is {word_count_references_avg}")
    print("--------------------------------")

    result = {
        "number_of_data": len(predictions),
        "rouge": rouge_results,
        "bleu": bleu_results,
        "meteor": meteor_results,
        "word_count_predictions_avg": word_count_predictions_avg,
        "word_count_references_avg": word_count_references_avg,
        "top_n": n,
        "worst_n_entries(rougeL)": worst_n_entries,
        "best_n_entries(rougeL)": best_n_entries
    }

    # save the results
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)