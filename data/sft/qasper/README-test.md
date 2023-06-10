Dataset of Information Seeking Questions and Answers Anchored in Research Papers: Test Set and Evaluator
--------------------------------------------------------------------------------------------------------                                      

## Version: 0.3

The tarball you found this file in should contain the test split of the Qasper dataset version 0.3 and the official evaluator script.

Please make sure you access the test file only to evaluate your finalized model.

## Images of tables and figures

You can download them here: https://qasper-dataset.s3.us-west-2.amazonaws.com/test_figures_and_tables.tgz

## Evaluation

You can evaluate your model using the stand alone evaluator as follows:

```
python qasper_evaluator.py --predictions predictions.jsonl --gold qasper-test-v0.3.json [--text_evidence_only]
```

Run the following to understand the arguments

```
python qasper_evaluator.py -h
```
