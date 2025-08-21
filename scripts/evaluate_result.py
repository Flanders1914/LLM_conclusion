# -*- coding: utf-8 -*-
# This script is used to evaluate the results of the model's predictions
# It evaluates the following metrics:
# 1. ROUGE(both individual and overall)
# 2. BLEU(only overall)
# 3. Meteor(both individual and overall)
# 4. Word count(both individual and overall)
# The output is saved in a jsonl file, the first line is the overall results, the rest are the individual results

import evaluate
import argparse
import os
import json
import nltk

# download the nltk data if not exist
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4",  quiet=True)
nltk.download("punkt", quiet=True) 

def evaluate_rouge(predictions, references, rouge):
    results = rouge.compute(predictions=predictions, references=references)
    return results

def evaluate_bleu(predictions, references, bleu):
    # convert the references to the format of [[r]]
    references = [[r] for r in references]
    results = bleu.compute(predictions=predictions, references=references)
    return results

def evaluate_meteor(predictions, references, meteor):
    results = meteor.compute(predictions=predictions, references=references)
    return results

def evaluate_rouge_individual(prediction, reference, rouge):
    """Evaluate ROUGE score for a single prediction-reference pair"""
    results = rouge.compute(predictions=[prediction], references=[reference])
    return results

def evaluate_meteor_individual(prediction, reference, meteor):
    results = meteor.compute(predictions=[prediction], references=[reference])
    return results

def evaluate_word_count_individual(prediction, reference):
    # return the word count of prediction and reference
    return len(prediction.split()), len(reference.split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # create the output directory if it doesn't exist
    dirpath = os.path.dirname(args.output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    predictions = []
    references = []
    individual_results = []

    # Read the input file, data["output"] is the prediction, data["answer"] is the reference, data["input"] is the model input/prompt
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            data = json.loads(line)
            # get the predictions and references
            prediction = data["output"]
            reference = data["answer"]
            model_input = data["input"]
            predictions.append(prediction)
            references.append(reference)
            
            # Store original data for individual analysis
            individual_results.append({
                "line_number": line_num,
                "input": model_input,
                "prediction": prediction,
                "reference": reference
            })

    # evaluate the results
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    print(f"Start evaluating the individual results in {args.input_path}")
    print("--------------------------------")
    print(f"The number of data entries is {len(predictions)}")
    print("--------------------------------")
    total_word_count_predictions = []
    total_word_count_references = []

    # evaluate the individual results
    for i, result in enumerate(individual_results):
        if i % 100 == 0:
            print(f"Evaluated {i} entries")
        rouge_results = evaluate_rouge_individual(result["prediction"], result["reference"], rouge)
        meteor_results = evaluate_meteor_individual(result["prediction"], result["reference"], meteor)
        word_count_prediction, word_count_reference = evaluate_word_count_individual(result["prediction"], result["reference"])
        result["rouge"] = rouge_results
        result["meteor"] = meteor_results
        result["word_count_prediction"] = word_count_prediction
        result["word_count_reference"] = word_count_reference
        total_word_count_predictions.append(word_count_prediction)
        total_word_count_references.append(word_count_reference)
    print("Completed evaluating the individual results")
    print("--------------------------------")

    # Guard against empty inputs
    if len(predictions) == 0:
        print("[ERROR] No prediction/reference pairs found in input. Exiting without writing results.")
        raise SystemExit(1)

    # evaluate the overall results
    print("Evaluate the overall results")
    print("--------------------------------")
    rouge_results = evaluate_rouge(predictions, references, rouge)
    print(f"The overall rouge results are: ROUGE-L: {rouge_results['rougeL']:.4f}, ROUGE-L-Sum: {rouge_results['rougeLsum']:.4f}, ROUGE-1: {rouge_results['rouge1']:.4f}, ROUGE-2: {rouge_results['rouge2']:.4f}")
    print("--------------------------------")
    bleu_results = evaluate_bleu(predictions, references, bleu)
    print(f"The overall bleu results are: BLEU: {bleu_results['bleu']:.4f}, Unigram: {bleu_results['precisions'][0]:.4f}, Bigram: {bleu_results['precisions'][1]:.4f}, Trigram: {bleu_results['precisions'][2]:.4f}, Quadgram: {bleu_results['precisions'][3]:.4f}, Brevity Penalty: {bleu_results['brevity_penalty']:.4f}")
    print("--------------------------------")
    meteor_results = evaluate_meteor(predictions, references, meteor)
    print(f"The overall meteor results are: {meteor_results['meteor']:.4f}")
    print("--------------------------------")
    word_count_predictions_avg = sum(total_word_count_predictions) / len(total_word_count_predictions)
    print(f"The average word count of predictions is {word_count_predictions_avg}")
    word_count_references_avg = sum(total_word_count_references) / len(total_word_count_references)
    print(f"The average word count of references is {word_count_references_avg}")
    print("--------------------------------")

    result = {
        "number_of_data": len(predictions),
        "rouge": rouge_results,
        "bleu": bleu_results,
        "meteor": meteor_results,
        "word_count_predictions_avg": word_count_predictions_avg,
        "word_count_references_avg": word_count_references_avg,
    }

    print("Completed evaluating the overall results")
    print("--------------------------------")
    print(f"Saving the results to {args.output_path} in jsonl format")
    print("The first line is the overall results, the rest are the individual results")
    # save the results
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        for individual_result in individual_results:
            f.write(json.dumps(individual_result, ensure_ascii=False) + "\n")
    print("--------------------------------")