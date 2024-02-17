# CEV-LM: Controlled Edit Vector Language Model for Shaping Natural Language Generations

Source code accompanying our paper, "CEV-LM: Controlled Edit Vector Language Model for Shaping Natural Language Generations".

**Authors:** Anonymous for EMNLP 2023 Submission 

## Environment Installation

Run the following commands to install the environment for CEV-LM. Note that you will have to change the `<PATH_TO_ENV>` in `environment.yml` to wherever you want to store the environment. Altneratively, you can use `requirements.txt`.

```bash
conda create -n nedit python=3.6
conda activate nedit
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz --no-deps
conda env update -n nedit -f environment.yml
cd third-party/gtd/
pip install -r requirements.txt

cd ../..
python -m nltk.downloader punkt
module load openblas # use this if blas is not loaded by default
pip install pyhocon IPython tensorboard_logger line_profiler nltk==3.2.3 bert_score
conda install pytorch==1.2.0
```

## Running the code

Before running any scripts, you must run the following snippit to ensure all environment variables are set:

```bash
cd cev-lm
conda activate nedit
export TEXTMORPH_DATA=$(pwd -P)
export PYTHONPATH=.
```

You must also install [GloVE word vectors](https://nlp.stanford.edu/projects/glove/) and place them in a directory called `word_vectors`. We use `glove.6B.300d_yelp` from this [notebook](https://worksheets.codalab.org/worksheets/0xa915ba2f8b664ddf8537c83bde80cc8c/).

You should also install the `yelpdata` from the same [notebook](https://worksheets.codalab.org/worksheets/0xa915ba2f8b664ddf8537c83bde80cc8c/). You can download the train, test, and validation `.tsv` files from this [bundle](https://worksheets.codalab.org/worksheets/0xa915ba2f8b664ddf8537c83bde80cc8c?bundle=0x984fe19b60f1479b925933eacbfda8d8&focus=9&subfocus=0). Make sure to download `free.txt` as well. These should be placed in a directory (`<DATA_DIR>`).

You should also check `cev-lm/textmorph/data.py` and ensure the paths are acceptable.

### To prepare the data

Run `python corpus/filter_attribute.py <FEATURE> <DELTA> <TOL>` (note to change `do_evaluation` flag in script to `True`)
Alternatively, run `scripts/create_datasets.sh` to do it automatically. You will need to change the feature parameter.

### To train
Change `configs/edit_model/edit_baseline.txt` with your desired settings. Likely, you only need to change `perturb_val` and `path`.
Run `python textmorph/edit_model/main.py configs/edit_model/edit_baseline.txt`

Alternatively, if you are on a server, you can run `train.slurm`. It is set up for SLURM and you may need to change the directories/settings.

### To evaluate
Run `python textmorph/edit_model/main.py <EXP_ID>` with the experiment id to generate predictions. You can find the experiment id as the experiment folder name in `edit_runs`. You should find a file called `test_preds.txt` in the experiment folder. 

We choose to move the test preds to the respective folder for the dataset in a subfolder called `preds`. We prefix the file with the value used for perturbation. For example:

```
cev-lm
└───data
│   └───speed
|   │   └───data_0.5_0.1
|   |   │   |   test.tsv
|   |   │   |   train.tsv
|   |   │   |   valid.tsv
|   |   │   |   free.txt
|   |   │   └───preds
|   |   │   |   |   0.0_test_preds.txt
|   |   │   |   |   0.5_test_preds.txt
```

You can then run `python corpus/evaluate_attribute.py <MODEL>`. This file can also be used for all baselines (`ssd-lm, mucoco, gpt3, prefix`). You may need to set the directories in the file accordingly.

## Common Bugs and Fixes

### To get metrics
Run `python corpus/filter_attribute.py` (note to change `do_evaluation` flag in script to `True`)
Note that you need `nltk >= 3.4.1` to get METEOR score

### Evaluation Bug
JSONDecodeError -> replace `nan` with `0.0` in metadata.txt

