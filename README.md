# CEV-LM: Controlled Edit Vector Language Model for Shaping Natural Language Generations

Source code accompanying our paper, "CEV-LM: Controlled Edit Vector Language Model for Shaping Natural Language Generations".

If this code helps you in your research, please cite the following publication (BibTeX at the bottom of the README):

> Samraj Moorjani, Adit Krishnan, and Hari Sundaram. 2024. CEV-LM: Controlled Edit Vector Language Model for Shaping Natural Language Generations. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1325–1340, St. Julian’s, Malta. Association for Computational Linguistics.

## Directory Structure

```
emnlp2023-code
│   README.md
│
└───cev-lm
│   │   environment.yml
│   │   README.md
|   |   requirements.txt
│   │
│   └───configs             # configs for CEV-LM
│   └───corpus              # data preparation + evaluation
│   └───figure_generation   # scripts to generate figures
│   └───gtd                 
│   └───scripts             # scripts
│   └───textmorph           # contains code for prototype-then-edit
│   └───third-party         
│   
└───baseline_file
|   └───mucoco
|   |   decode_examples.sh    
|   |   sts_gpt_train.py    
|   └───gpt
|   |   run_gpt.py          # run gpt baseline
|   └───prefix
|   |   txt_to_src_tgt.py   # convert .txt to .src and .tgt
|
└───attribute_control
    │   data_utils.py
    |   README.md
    │   pairwise_data_gen.py
    │   attribute_regressor.py
    |   
    └───scripts             # scripts

```

`cev-lm` contains all necessary code to reproduce our results for the CEV-LM architecture.

`attribute_control` contains code for training the discriminators/classifiers used in the baselines (SSD-LM and MuCoCO).

## Environment Installation + Running Code

See `cev-lm/README.md` for instructions of how to set up the environment for CEV-LM and do training/evaluation.

See `attribute_control/README.md` for instructions of how to set up the environment for the classifiers/discriminators for the baselines along with training instructions.

## Baselines 

The following sections contain information about running the baselines. We do not include code from their repositories, but include any code we feel helpful to pulling up baselines.

### SSD-LM

The SSD-LM [repo](https://github.com/xhan77/ssd-lm) contains a [notebook](https://colab.research.google.com/drive/1vNKqvzzJQp3k89QPuns5ibsq-VNC9wGN?usp=sharing) that we use to run all tests. Simply upload the classifier model and change the respective hyperparameters in the cell labeled "Controlled Generation".

### MuCoCO

Similarly, the MuCoCO [repo](https://github.com/Sachin19/mucoco) contains all information needed to set up the environment and run MuCoCO. We found the setup to be a bit confusing, so we included the main file you need: `benchmarks/mucoco/decode_example.sh`. There is a dependency on a `STS_MODEL` which can be trained using the `sts_gpt_train.py` script we included.

### Prefix Tuning

The Prefix Tuning [repo](https://github.com/XiangLi1999/PrefixTuning) contains all information needed to set up and run the Prefix Tuning benchmark. 

We include `benchmarks/prefix/txt_to_src_tgt.py` to help convert the data to the format required in the repository. 

Note that you should use `seq2seq/train_bart.py` to train the benchmark. You may need to change the filepaths to accomodate.

### GPT-3

We include a file to help you start running the baseline in `benchmarks/gpt/run_gpt.py`. All information regarding prompt creation is included in the paper. You will need to install the OpenAI API using `pip install openai` and set the API key as an environment variable. Find more information [here](https://platform.openai.com/docs/quickstart/add-some-examples).

## Citation
If this code helps in your research, please cite our work. The BibTeX citation is as follows:

```
@inproceedings{moorjani-etal-2024-cev,
    title = "{CEV}-{LM}: Controlled Edit Vector Language Model for Shaping Natural Language Generations",
    author = "Moorjani, Samraj  and
      Krishnan, Adit  and
      Sundaram, Hari",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.80",
    pages = "1325--1340",
    abstract = "As large-scale language models become the standard for text generation, there is a greater need to tailor the generations to be more or less concise, targeted, and informative, depending on the audience/application. Existing control approaches primarily adjust the semantic (e.g., emotion, topics), structural (e.g., syntax tree, parts-of-speech), and lexical (e.g., keyword/phrase inclusion) properties of text, but are insufficient to accomplish complex objectives such as pacing which control the complexity and readability of the text. In this paper, we introduce CEV-LM - a lightweight, semi-autoregressive language model that utilizes constrained edit vectors to control three complementary metrics (speed, volume, and circuitousness) that quantify the shape of text (e.g., pacing of content). We study an extensive set of state-of-the-art CTG models and find that CEV-LM provides significantly more targeted and precise control of these three metrics while preserving semantic content, using less training data, and containing fewer parameters.",
}
```
