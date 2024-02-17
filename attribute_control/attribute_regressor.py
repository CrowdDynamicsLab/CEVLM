import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torchvision import transforms

from training_utils import get_args
from data_utils import SpeedDataset, get_dataset, ShuffleChunkTransform, TokenizeTransform

import evaluate
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer

class Switch(dict):
    def __getitem__(self, item):
        for key in self.keys():                 # iterate over the intervals
            # if item in key:                     # if the argument is in that interval
            if key[0] <= item < key[1]:
                return super().__getitem__(key) # return its associated value
        raise KeyError(item)                    # if not in any interval, raise KeyError

def get_default_switch(values, num_bins=10):

    values.sort()
    hist, bin_edges = np.histogram(np.array(values), bins=num_bins)
    bin_midpoints = np.add(bin_edges[:-1], np.diff(bin_edges) / 2)
    # reverse_switch = {i : (bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_bins)}
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    range_dict = {(bin_edges[i], bin_edges[i+1]) : i for i in range(num_bins)}
    
    default_switch = Switch(range_dict)
    
    return default_switch, bin_midpoints

class CustomCLFLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.midpoints = kwargs.pop('midpoints', None)
        self.midpoints = torch.tensor(self.midpoints).reshape(1, -1).to("cuda")
        self.loss_fct = nn.MSELoss()
        super(CustomCLFLossTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").double()
        del inputs['labels']

        outputs = model(**inputs)
        probs = torch.softmax(outputs.get('logits'), dim=1)
        feat_pred = (probs * self.midpoints).sum(dim=1)
        loss = self.loss_fct(feat_pred.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss

def main(args, predict_difference=True, use_custom_loss=True):
    device = torch.device("cuda")

    print('Loading data...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.model_name == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token

    if not args.data_dir:   
        if predict_difference:
            data_dir = f'./data/yelpdata_{args.feat}s_deltas_1M.txt'
        else:
            data_dir = f'./data/yelpdata_{args.feat}s_1000000.txt'

    sentences_with_feats = get_dataset(data_dir)[:100000]
    values = [float(s[-1]) for s in sentences_with_feats]

    # Get the indices of NaN values in the values list
    nan_indices = [i for i, v in enumerate(values) if np.isnan(v)]

    # Remove the elements at the NaN indices from both lists
    sentences_with_feats = [s for i, s in enumerate(sentences_with_feats) if i not in nan_indices]
    values = [v for i, v in enumerate(values) if i not in nan_indices]

    do_regression = args.num_classes == 1
    switch, midpoints = (None, None) if do_regression else get_default_switch(values, num_bins=args.num_classes)
    print(switch)

    switch = None if use_custom_loss else switch 

    print('Creating datasets...')
    dataset = SpeedDataset(
        sentences_with_feats,
        tokenizer,
        predict_difference=predict_difference,
        transform=transforms.Compose([
            ShuffleChunkTransform(args.chunk_len),
            TokenizeTransform(tokenizer)
        ]),
        switch=switch
    )

    test_size = 1 - args.train_test_split # e.g. 20%

    # Calculate the sizes of the train and test sets
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size

    # Split the dataset into train and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # config = AutoConfig.from_pretrained(args.model_name, num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_classes)
    if args.model_name == 'gpt2':
        model.config.pad_token_id = model.config.eos_token_id

    model_name = f'trainer_checkpoint_{"reg" if do_regression else "clf"}_{args.num_epochs}_{predict_difference}_{use_custom_loss}_{args.lr}_{args.batch_size}_{args.train_test_split}{"_" + args.notes if args.notes else ""}'
    Path(f'./{args.feat}_control').mkdir(parents=True, exist_ok=True)
    save_dir = f'./{args.feat}_control/{args.model_name.replace("facebook/", "").split("-")[0]}/{model_name}'

    acc_metric = evaluate.load("accuracy")
    mae_metric = evaluate.load("mae")
    def compute_regression_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits[0] if 'bart' in args.model_name else logits
        return mae_metric.compute(predictions=predictions, references=labels)

    def compute_classification_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits[0] if 'bart' in args.model_name else logits, axis=-1)
        accuracy = acc_metric.compute(predictions=predictions, references=labels)
        mae = mae_metric.compute(predictions=predictions, references=labels)
        return {**accuracy, **mae}

    def compute_custom_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits[0] if 'bart' in args.model_name else logits
        probs = np.array(torch.softmax(torch.tensor(predictions), dim=1))
        feat_pred = (probs * midpoints).sum(axis=1)
        return mae_metric.compute(predictions=feat_pred, references=labels)

    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=int(args.batch_size / torch.cuda.device_count()),
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        # evaluation
        evaluation_strategy="steps",
        eval_steps=1000,
        eval_accumulation_steps=16,
        # logging to tensorboard
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=500,
        # saving
        save_total_limit = 2,
        save_strategy = "no",
    )

    trainer = None
    if use_custom_loss:
        trainer = CustomCLFLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_custom_metrics,
            midpoints=midpoints,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_regression_metrics if do_regression else compute_classification_metrics,
        )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    args = get_args()
    main(args)