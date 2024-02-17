# Attribute Classifier/Regressor for SSD-LM/MuCoCO

Source code accompanying our paper, "CEV-LM: Controlled Edit Vector Language Model for Shaping Natural Language Generations". This section addresses the classifier and regressor used to the benchmark models.

## Environment Setup

You can setup the environment by running `pip install -r requirements.txt`.

## Running Instructions

### Data Generation

After setting up the environment, you can get the data as `yelpdata.json` from this [notebook](https://worksheets.codalab.org/worksheets/0xa915ba2f8b664ddf8537c83bde80cc8c/). You should then run `scripts/yelp_json_to_txt.py` to convert it to a `.txt` file.

You will need to run `cev-lm/corpus/compute_attribute.py` for each attribute.

Then you can run `python pairwise_data_gen.py` to create feature pairs. You may need to adjust the filepaths. If you are running on a machine, we also provide a slurm script in `scripts/compute_data_pairs.slurm`.

### Model Training

To train the model, you should run `python attribute_regressor.py` with the appropriate hyperparameters. You can see the config in `training_utils.py`. Some important features to pay attention to include `--data_dir` to specify the data file, `--regression` as a flag of whether to train the regressor (if left empty, train classifier), and `--feat` to specify the feature. 

For convenience, we also provide a slurm script in `scripts/train.slurm`.

## Common Errors

### Faiss assertion err == CUBLAS_STATUS_SUCCESS failed
- Install from conda instead of pip using `conda install -c pytorch faiss-gpu`

### RuntimeError: module compiled against API version 0xe but this version of numpy is 0xd
- Upgrade numpy
- Was 1.22.1 before, upgraded to 1.21

