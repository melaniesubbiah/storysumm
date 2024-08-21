# STORYSUMM: Evaluating Faithfulness in Story Summarization

Code and data for the paper: https://arxiv.org/pdf/2407.06501

### Dataset
The StorySumm dataset is in the file ``storysumm.json``

Description of data fields:\
``label`` - final label for the summary (0 is unfaithful, 1 is faithful)\
``difficulty`` - easy or hard\
``story`` - the source story\
``summary`` - the LLM-generated summary, split into sentences\
``errors`` - sentence-level inconsistency labels\
``story-id`` - identifier for the source story\
``explanations`` - a reason for each 0 in the errors list, listed in order\
``claims`` - atomic claims made in the summary, generated and listed by GPT-4\
``split`` - val or test\
``model`` - LLM used to generate the summary\

### Install

```shell
pip install -r requirements.txt
```

Within the ``evaluators`` directory, clone and install [MiniCheck](https://github.com/Liyan06/MiniCheck), [UniEval](https://github.com/maszhongming/UniEval), and [AlignScore](https://github.com/yuh-zha/AlignScore). The installs for each are not necessarily compatible with each other, so you may need to use separate virtual environments. Refer to the original repos for any code updates and install instructions.

### Evaluators
Generated predictions from the different evaluators can be found in ``evaluators/predicted_labels``. The code to run the different evaluators is in the ``evaluators`` directory. First, fill in your access credentials for the relevant APIs in ``evaluators/api_secrets.py``.

Description of data fields in predicted labels (files have different combinations of fields):\
``label`` - predicted label for the summary\
``sentence_labels`` - predicted labels for each sentence in the summary\
``claim_labels`` - predicted labels for each claim in teh summary\
``probs`` - the probability of a 1 label (either at the summary or sentence level) or the explanation given by a CoT method

Running evaluators (check the top of each script for additional parameters):\
``llm_evaluators.py`` - LLM evaluators, use --model and --mode to run different settings\
``fables.py`` - FABLES\
``minicheck.py`` - MiniCheck\
``alignscore.py`` - AlignScore\
``unieval.py`` - UniEval

You can then check results on the full expanded set of labels using the ``Results.ipynb`` notebook.

### Results (8/22/24 Update)

| Split    | Method           | Cohen's Kappa | % Faithful | Precision | Recall    | % Easy    | % Hard    | Balanced Accuracy |
|----------|------------------|---------------|------------|-----------|-----------|-----------|-----------|-------------------|
| Val/Test | UniEval          | 0.25/0.04     | 39/30      | 0.38/0.47 | 0.62/0.32 | 80.0/80.0 | 60.0/68.0 | 65.3/51.8         |
| Val/Test | AlignScore       | 0.21/-0.07    | 42/68      | 0.36/0.42 | 0.62/0.64 | 80.0/40.0 | 53.3/24.0 | 63.3/46.4         |
| Full     | Binary (Claude-3) | 0.06          | 95         | 0.40      | 1.00      | 20.0      | 2.5       | 54.2              |
| Full     | Binary (GPT-4)   | 0.11          | 70         | 0.42      | 0.78      | 55.0      | 25.0      | 56.4              |
| Full     | Binary (Mixtral) | 0.12          | 91         | 0.41      | 1.00      | 15.0      | 15.0      | 57.5              |
| Full     | CoT (Claude-3)   | 0.10          | 90         | 0.41      | 0.97      | 25.0      | 10.0      | 56.1              |
| Full     | CoT (GPT-4)      | 0.08          | 94         | 0.40      | 1.00      | 25.0      | 2.5       | 55.0              |
| Full     | CoT (Mixtral)    | 0.04          | 97         | 0.39      | 1.00      | 0.0       | 7.5       | 52.5              |
| Full | FABLES | 0.33          | 55         | 0.53      | 0.78      | 70.0      | 52.5      | 68.1              |
| Full     | MiniCheck        | 0.02          | 16         | 0.4       | 0.17      | 90.0      | 82.5      | 50.8              |