import json
import requests
import time
import argparse
import os

def evaluate_conclusion_pair(api_key, candidate_conclusion, reference_conclusion, model="gpt-3.5-turbo"):
    """
    Send a pair of conclusion to ChatGPT API and get similarity score
    """
    prompt = f"""
Please act as an impartial judge to evaluate the semantic equivalence between the Candidate conclusion and the Reference conclusion.
Candidate conclusion:
{candidate_conclusion}
Reference conclusion: 
{reference_conclusion}
On a precise scale from 0 to 100, how likely is it that the candidate conclusion is describing the same meaning as the reference conclusion? \
The output should be ONLY a JSON dict format including a key "reason" with a string value to explain your evaluation, and a key "score" with integer value between 0 and 100 as the final result.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that evaluates the similarity of two conclusions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    print(f"Making API request with model: {model}")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        print(f"Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")
        return {"score": 0, "reason": f"API error: {response.status_code}"}
    
    result = response.json()
    message_content = result["choices"][0]["message"]["content"]
    
    print(f"API response content: {message_content}")
    
    # Manually extract score from response
    try:
        # Try to parse JSON from the message
        score_dict = json.loads(message_content)
        return score_dict
    except json.JSONDecodeError:
        # If that fails, try to find the score in the text
        print("Could not parse response as JSON, attempting to extract score...")
        try:
            import re
            score_match = re.search(r'["\']score["\']\s*:\s*(\d+)', message_content)
            reason_match = re.search(r'["\']reason["\']\s*:\s*["\'](.*?)["\']', message_content)
            
            if score_match:
                score = int(score_match.group(1))
                reason = reason_match.group(1) if reason_match else "No reason provided"
                return {"score": score, "reason": reason}
            else:
                print("Could not extract score from response")
                return {"score": 0, "reason": "Failed to parse response"}
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {"score": 0, "reason": "Failed to parse response"}

def main():
    parser = argparse.ArgumentParser(description='Evaluate conclusion similarity using ChatGPT')
    parser.add_argument('--input', type=str, required=True, help='Path to the JSONL file with conclusion pairs')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='OpenAI model to use')
    parser.add_argument('--max-pairs', type=int, default=None, help='Maximum number of conclusion pairs to evaluate')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls in seconds')
    
    args = parser.parse_args()
    
    # Input validation
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    if not args.api_key.startswith('sk-'):
        print("Warning: API key should typically start with 'sk-'")
    
    results = {
        "conclusion_evaluations": {},
        "summary": {"total_score": 0, "count": 0}
    }

    with open(args.input, 'r') as f:
        for index, line in enumerate(f, start=1):
            try:
                data_item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {index}: {e}")
                continue
                
            candidate = data_item.get('output', '')
            reference = data_item.get('answer', '')
            
            if not candidate or not reference:
                print(f"Warning: Missing candidate or reference for line {index}")
                continue
            
            # Add a delay between API calls to avoid rate limiting
            time.sleep(args.delay)
            
            try:
                similarity = evaluate_conclusion_pair(args.api_key, candidate, reference, args.model)
                
                # Store result
                results["conclusion_evaluations"][index] = {
                    "candidate": candidate,
                    "reference": reference,
                    "similarity": similarity
                }
                
                if "score" in similarity:
                    score = similarity["score"]
                    results["summary"]["total_score"] += score
                    results["summary"]["count"] += 1
                    print(f"For the {index}th pair: Score = {score}/100")
            except Exception as e:
                print(f"Error processing {index}th pair: {e}")
                # Continue with the next pair   
            if args.max_pairs is not None and index == args.max_pairs:
                print(f"Reached the maximum number of pairs to evaluate: {args.max_pairs}")
                break

    # Calculate average score before saving
    if results["summary"]["count"] > 0:
        avg_score = results["summary"]["total_score"] / results["summary"]["count"]
        results["summary"]["average_score"] = avg_score
    
    # Save the test result
    out_parent_dir = os.path.dirname(args.output)
    if not os.path.exists(out_parent_dir):
        os.makedirs(out_parent_dir)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {args.output}")
    if results["summary"]["count"] > 0:
        avg_score = results["summary"]["average_score"]
        print(f"Average CLAIR similarity score: {avg_score:.2f}")
        print(f"Evaluated {results['summary']['count']} conclusion pairs successfully.")

if __name__ == "__main__":
    main()