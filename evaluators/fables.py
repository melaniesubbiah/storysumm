import json
import openai
import argparse
from api_secrets import OA_KEY, OA_ORGANIZATION


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-4-turbo-preview')
args = parser.parse_args()

openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY

firstpart = "You are provided with a context and a statement. Your task is to carefully read the context and then determine whether the statement is true or false. Use the information given in the context to make your decision."

def chatgpt(messages, model, temperature=0, max_tokens=50):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    claim = response['choices'][0]['message']['content']
    return claim.strip()

with open('../storysumm.json', 'r') as f:
    gold_data = json.loads(f.read())

preds = {}
for key, gd in gold_data.items():
    allabs = []
    story = gd['story']
    claims = gd['claims']
    for claim in claims:
        messages = [
            {'role': 'user', 'content': firstpart},
            {'role': 'user', 'content': f'Context:\n{story}\n\nStatement:\n{claim}\n\nQuestion: Based on the context provided, is the above statement True or False?\n\nAnswer:'},
        ]
        answer = chatgpt(messages, model=args.model)
        if answer == 'True':
            allabs.append(1)
        else:
            allabs.append(0)
    preds[key] = {
        "claim_labels": allabs,
        "label": int(sum(allabs) == len(allabs))
    }
    
with open(f'predicted_labels/fables-{args.model}.json', 'w') as f:
    f.write(json.dumps(preds))
    
    