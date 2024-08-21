from minicheck.minicheck import MiniCheck
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='flan-t5-large')
args = parser.parse_args()

with open('../../storysumm.json', 'r') as f:
    gold_data = json.loads(f.read())

scorer = MiniCheck(model_name=args.model, cache_dir='./ckpts', enable_prefix_caching=False)

preds = {}
for key, gd in gold_data.items():
    docs = [gd['story']]*len(gd['summary'])
    pred_labels, raw_prob, _, _ = scorer.score(docs=docs, claims=gd['summary'])

    preds[key] = {
        'sentence_labels': pred_labels,
        'label': int(sum(pred_labels) == len(pred_labels)),
        'probs': raw_prob
    }

with open(f'../predicted_labels/minicheck-{args.model}.json', 'w') as f:
    f.write(json.dumps(preds))
