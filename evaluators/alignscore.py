from alignscore import AlignScore
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='roberta-large')
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda:1')
args = parser.parse_args()

model_paths = {
    'roberta-base' : 'https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt',
    'roberta-large' : 'https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt'
}

scorer = AlignScore(model=args.model, batch_size=args.bs, device=args.device, ckpt_path=model_paths[args.model], evaluation_mode='bin')

with open('../../../storysumm.json', 'r') as f:
    storysumm = json.loads(f.read())

contexts, claims, keys = [], [], []
for key, dat in storysumm.items():
    contexts.append(dat['story'])
    claims.append(' '.join(dat['summary']))
    keys.append(key)

scores = scorer.score(contexts=contexts, claims=claims)

preds = {}
for key, score in zip(keys, scores):
    preds[key] = {'probs': score}
breakpoint()
with open(f'../../predicted_labels/alignscore-{args.model}.json', 'w') as f:
    f.write(json.dumps(preds))
