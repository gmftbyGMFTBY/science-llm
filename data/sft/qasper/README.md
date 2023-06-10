A Dataset of Information Seeking Questions and Answers Anchored in Research Papers
----------------------------------------------------------------------------------

## Version 0.3

The tarball you found this README in should contain the training and development sets of Qasper version 0.3. The images of the tables and figures
in the papers associated can be found here: https://qasper-dataset.s3.us-west-2.amazonaws.com/train_dev_figures_and_tables.tgz

The full text of the papers is extracted from S2ORC (Lo et al., 2020).

Each file is in JSON format, where the keys are arxiv ids, and the values are dicts containing `title`, `abstract`, `full_text`, `figures_and_tables`, and `qas` (QA pairs).

## Differences from v0.2

Due to an issue in the annotation interface, a small number of annotations (about 0.6%) had multiple answer types (e.g.: unanswerable and boolean; see more information on answer types in the final section of this README) in v0.2. These were manually fixed to create v0.3. These fixes affected train, development, and test sets.

## Figures and tables

These are new starting version 0.2. The actual images of the figures and tables can be downloaded from the link above. The JSON files contain the
captions to those images in the `figure_and_table_captions` field.

This field is a dict whose keys are file names of the images of tables and figures, and the values are their captions.

For example, the paper with arxiv id `1811.00942` is in the training set, and contains the following `figures_and_tables` field:

```
"figures_and_tables": [
      {
        "file": "3-Table1-1.png",
        "caption": "Table 1: Comparison of neural language models on Penn Treebank and WikiText-103."
      },
      {
        "file": "4-Figure1-1.png",
        "caption": "Figure 1: Log perplexity\u2013recall error with KN-5."
      },
      {
        "file": "4-Figure2-1.png",
        "caption": "Figure 2: Log perplexity\u2013recall error with QRNN."
      },
      {
        "file": "4-Table2-1.png",
        "caption": "Table 2: Language modeling results on performance and model quality."
      }
]
``` 

and when you download the `train_dev_figures_and_tables` tarball, you will see four files in `train/1811.00942`, with file names corresponding to
the `file` fields in the list above.

## Fields specific to questions:

 - `nlp_background` shows the experience the question writer had. The values can be `zero` (no experience), `two` (0 - 2 years of experience), `five` (2 - 5 years of experience), and `infinity` (> 5 years of experience). The field may be empty as well, indicating the writer has chosen not to share this information.

 - `topic_background` shows how familiar the question writer was with the topic of the paper. The values are `unfamiliar`, `familiar`, `research` (meaning that the topic is the research area of the writer), or null.

 - `paper_read`, when specified shows whether the questionwriter has read the paper.

 - `search_query`, if not empty, is the query the question writer used to find the abstract of the paper from a large pool of abstracts we made available to them.

## Fields specific to answers

Unanswerable answers have `unanswerable` set to true. The remaining answers have exactly one of the following fields being non-empty.

 - `extractive_spans` are spans in the paper which serve as the answer.
 - `free_form_answer` is a written out answer.
 - `yes_no` is true iff the answer is Yes, and false iff the answer is No.

`evidence` is the set of paragraphs, figures or tables used to arrive at the answer. When the evidence is a table or a figure, it starts with the
string `FLOAT SELECTED`, and contains the caption of the corresponding table or figure.

`highlighted_evidence` is the set of sentences the answer providers selected as evidence if they chose textual evidence. The text in the `evidence` field is a mapping from these sentences to the paragraph level. That is, if you see textual evidence in the `evidence` field, it is guaranteed to be entire paragraphs, while that is not the case with `highlighted_evidence`.
