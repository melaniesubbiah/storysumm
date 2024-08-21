import argparse
import backoff
import json
import openai
import os
import numpy as np
import requests
from anthropic import Anthropic
from api_secrets import OA_KEY, OA_ORGANIZATION, CLAUDE_KEY, HF_AUTHORIZATION


openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY

anthropic = Anthropic()
anthropic.api_key = CLAUDE_KEY

hf_headers = {"Authorization": HF_AUTHORIZATION}
mixtral_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

with open('systemprompt.txt', 'r') as f:
    SYSTEM_PROMPT = f.read().strip()
QUESTION = "Is all of the information in the summary consistent with the story? Ignore summary sentences that are just commentary/interpretation. You should answer Yes or No.\n"


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def gpt(messages, model, temperature, max_tokens=650):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, logprobs=True, top_logprobs=2
    )
    label = response['choices'][0]['message']['content']
    if label.startswith('Yes'):
        label = 1
    elif label.startswith('No'):
        label = 0
    logprobs = {}
    for answer in response.choices[0].logprobs.content[0].top_logprobs:
        if answer.token == 'Yes':
            logprobs[1] = np.round(np.exp(answer.logprob)*100, 2)
        elif answer.token == 'No':
            logprobs[0] = np.round(np.exp(answer.logprob)*100, 2)
    return label, logprobs[label]

def claude(system, messages, model, temperature, max_tokens=650):
    response = anthropic.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        temperature=temperature
    )
    output = response.content[0].text
    label = output.split('<answer>')[1].split('</answer>')[0]
    label = 1 if label == 'Yes' else 0
    return label, None


def mixtral(instruction, max_tokens=650):
    payload = {"inputs": instruction, "parameters":{"do_sample": False, "return_full_text": False, "max_new_tokens": max_tokens}}
    output = requests.post(mixtral_url, headers=hf_headers, json=payload).json()[0]['generated_text'].strip()
    label = output.split('<answer>')[1].split('</answer>')[0]
    label = 1 if label == 'Yes' else 0
    return label, None


def jq_label(story, summary, model, temperature, max_tokens=650):
    core_prompt = f'Story:\n{story}\n\nSummary:\n{summary}'
    answer_tags_prompt = 'Place your answer between <answer></answer> tags.'
    if model.startswith('claude'):
        messages = [
            {'role': 'user',
             'content': f'{core_prompt}\n\n{QUESTION}{answer_tags_prompt}'},
        ]
        label, confidence = claude(system=SYSTEM_PROMPT, messages=messages, model=model, temperature=temperature,
                                   max_tokens=max_tokens)
    elif model == 'mixtral':
        instruction = f"[INST] {SYSTEM_PROMPT}\n\n{core_prompt}\n\n{QUESTION}{answer_tags_prompt} [/INST]"
        label, confidence = mixtral(instruction=instruction, max_tokens=max_tokens)
    else:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': core_prompt},
            {'role': 'user', 'content': QUESTION}
        ]
        label, confidence = gpt(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
    return label, confidence

def cot_label(story, summary, model, temperature, max_tokens):
    core_prompt = f'Story:\n{story}\n\nSummary:\n{summary}'
    cot_prompt = 'Consider whether there are any details in the summary that are inconsistent with the story and provide a couple sentences of reasoning for why the summary is or is not consistent with the story.'
    followup_q = 'So overall, are all of the details in the summary consistent with the story? You should answer Yes or No.'
    answer_tags_prompt = 'Place your answer between <answer></answer> tags.'
    if model.startswith('claude'):
        messages = [
            {'role': 'user', 'content': f'{core_prompt}\n\n{cot_prompt}'},
        ]
        response = anthropic.messages.create(
            model=model,
            max_tokens=350,
            system=SYSTEM_PROMPT,
            messages=messages,
            temperature=0
        )
        explanation = response.content[0].text
        messages.append({'role': 'assistant', 'content': explanation})
        messages.append({'role': 'user', 'content': f'{followup_q} {answer_tags_prompt}'})
        label, confidence = claude(system=SYSTEM_PROMPT, messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
        confidence = explanation
    elif model == 'mixtral':
        instruction = f"[INST] {SYSTEM_PROMPT}\n\n{core_prompt}\n\n{cot_prompt} [/INST]"
        payload = {"inputs": instruction, "parameters":{"do_sample": False, "return_full_text": False, "max_new_tokens": 350}}
        explanation = requests.post(mixtral_url, headers=hf_headers, json=payload).json()[0]['generated_text']
        instruction = "<s>" + instruction + explanation + "</s>" + f" [INST] {followup_q} {answer_tags_prompt} [/INST]"
        label, confidence = mixtral(instruction=instruction, max_tokens=max_tokens)
        confidence = explanation.strip()
    else:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': core_prompt},
            {'role': 'user', 'content': cot_prompt}
        ]
        response = openai.ChatCompletion.create(
            model=model, messages=messages, temperature=0, max_tokens=350
        )
        explanation = response['choices'][0]['message']['content']
        messages.append({'role': 'assistant', 'content': explanation})
        messages.append({'role': 'user', 'content': followup_q})
        label, confidence = gpt(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
        confidence = explanation
    return label, confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Script to rate faithfulness.')
    parser.add_argument('--model', default='gpt-3.5-turbo-0125', choices=[
        'gpt-3.5-turbo-0125',
        'gpt-4-0125-preview',
        'claude-3-opus-20240229',
        'mixtral',
        'gpt-4o'
    ])
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--max_tokens', default=10, type=int)
    parser.add_argument('--outfilesuffix', default='', type=str)
    parser.add_argument('--mode', default='justquestion', choices=[
        'justquestion',
        'cot'
    ])

    args = parser.parse_args()
    
    if not os.path.exists(f"predicted_labels/{args.model}"):
        os.makedirs(f"predicted_labels/{args.model}")

    model_scores = {}
    if os.path.exists(os.path.join(f"predicted_labels/{args.model}", f"{args.mode}{args.outfilesuffix}.json")):
        with open(os.path.join(f"predicted_labels/{args.model}", f"{args.mode}{args.outfilesuffix}.json"), 'r') as f:
            model_scores = json.loads(f.read())

    with open("../storysumm.json", 'r') as f:
        storysumm = json.loads(f.read())

    for key, dat in storysumm.items():
        print(f"Summary {key}")
        if key in model_scores:
            continue
        text = dat['story'].strip()
        summary = ' '.join(dat['summary'])
        if args.mode == 'justquestion':
            label, confidence = jq_label(text, summary, args.model, args.temperature, args.max_tokens)
        elif args.mode == 'cot':
            label, confidence = cot_label(text, summary, args.model, args.temperature, args.max_tokens)
        model_scores[key] = {'label': label}
        if confidence:
            model_scores[key]['probs'] = confidence

        with open(os.path.join(f'predicted_labels/{args.model}', f"{args.mode}{args.outfilesuffix}.json"), "w") as f:
            f.write(json.dumps(model_scores))

