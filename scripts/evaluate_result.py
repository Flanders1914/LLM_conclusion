# python scripts/evaluate_result.py --input_path output/llama3/rct_dev.jsonl --output_path output/llama3/rct_dev_results.json
# python scripts/evaluate_result.py --input_path output/llama3/non_rct_dev.jsonl --output_path output/llama3/non_rct_dev_results.json
# python scripts/evaluate_result.py --input_path output/qwen3/non_rct_dev.jsonl --output_path output/qwen3/non_rct_dev_results.json
# python scripts/evaluate_result.py --input_path output/qwen3/rct_dev.jsonl --output_path output/qwen3/rct_dev_results.json
import evaluate
import argparse
import os
import json
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")

def evaluate_rouge(predictions, references, rouge):
    references = [[r] for r in references]
    results = rouge.compute(predictions=predictions, references=references)
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
    args = parser.parse_args()

    # create the output directory if it doesn't exist
    dirpath = os.path.dirname(args.output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    predictions = []
    references = []

    # read the input file
    with open(args.input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            # get the predictions and references
            predictions.append(data["output"])
            references.append(data["answer"])

    # evaluate the results
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    print(f"Start evaluating the results in {args.input_path}")
    print("--------------------------------")
    print(f"The number of data is {len(predictions)}")
    print("--------------------------------")
    print("Evaluate the results")
    print("--------------------------------")
    rouge_results = evaluate_rouge(predictions, references, rouge)
    print(f"The rouge results are {rouge_results}")
    print("--------------------------------")
    bleu_results = evaluate_bleu(predictions, references, bleu)
    print(f"The bleu results are {bleu_results}")
    print("--------------------------------")
    meteor_results = evaluate_meteor(predictions, references, meteor)
    print(f"The meteor results are {meteor_results}")
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
    }

    # save the results
    with open(args.output_path, "w") as f:
        json.dump(result, f)