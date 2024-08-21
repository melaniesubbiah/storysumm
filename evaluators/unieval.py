import json
from utils import convert_to_json
from metric.evaluator import get_evaluator

with open('../../storysumm.json', 'r') as f:
    storysumm = json.loads(f.read())

contexts, claims, keys = [], [], []
for key, dat in storysumm.items():
    contexts.append(dat['story'])
    claims.append(' '.join(dat['summary']))
    keys.append(key)

data = convert_to_json(output_list=claims,
                       src_list=contexts, ref_list=claims)
evaluator = get_evaluator('summarization')
scores = evaluator.evaluate(data)

preds = {}
for key, score in zip(keys, scores):
    preds[key] = {'probs': score['consistency']}

with open('../predicted_labels/unieval.json', 'w') as f:
    f.write(json.dumps(preds))
