# [NLQxform: A Language Model-based Question to SPARQL Transformer](https://ceur-ws.org/Vol-3592/paper2.pdf)

Ruijie Wang, Zhiruo Zhang, Luca Rossetto, Florian Ruosch, and Abraham Bernstein

:trophy: Winner of the [DBLP_QuAD KGQA Task - Scholarly QALD Challenge](https://kgqa.github.io/scholarly-QALD-challenge/2023/) at [The 22nd International Semantic Web Conference (ISWC 2023)](https://iswc2023.semanticweb.org/).

:boom: Based on NLQxform, we developed [NLQxform-UI](https://arxiv.org/abs/2403.08475) &mdash; an easy-to-use web-based interactive QA system over [DBLP Knowledge Graph](https://blog.dblp.org/tag/knowledge-graph/), which is also [open-sourced](https://github.com/ruijie-wang-uzh/NLQxform-UI).

:exclamation: Please note that, in the previous versions, the SSL certificate verification was disabled in `do_query.py` and `generator_main.py` due to the `SSL: CERTIFICATE_VERIFY_FAILED` error that we encountered before when querying the [SPARQL endpoint](https://dblp-kg.ltdemos.informatik.uni-hamburg.de/sparql).
We have removed this SSL error bypassing in the current version, as it is INSECURE and not necessary anymore.

----

## Environment Setup

Please set up a [Python](https://www.python.org/) environment with 
[Pytorch](https://pytorch.org/), 
[Pands](https://pandas.pydata.org/),
[Transformers](https://huggingface.co/docs/transformers/index),
[Sacrebleu](https://pypi.org/project/sacrebleu/),
[Rouge-score](https://pypi.org/project/rouge-score/),
[SPARQLWrapper](https://sparqlwrapper.readthedocs.io/en/latest/),
[Colorama](https://pypi.org/project/colorama/),
[Numpy](https://numpy.org/),
and [Beautifulsoup4](https://pypi.org/project/beautifulsoup4/) installed.

----

## Data and Models

Please download the data and models from [OSF](https://osf.io/k5mdg/?view_only=1e5bea63dc6f49aca0382b9444f3375b). (Unzip `data.zip` and `logs.zip` and move it to the root directory.)

----

## Inference using our Pre-trained Models

You can use the following command to load our pre-trained model and ask questions interactively.

```shell
# add `--verbose` to check all intermediate results, including entity linking results and generated final queries
python -u generator_main.py --resume_prefix v2 --device 0
```

A snapshot of our system answering the question *please enumerate other papers published by the authors of ‘BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding’* is shown below:

![overall results](https://github.com/ruijie-wang-uzh/NLQxform/blob/main/snapshot.png?raw=true)

----

## Training from Scratch

The following command can be used to train the model from scratch and evaluate it on the test set of the [DBLP-QuAD Challenge](https://codalab.lisn.upsaclay.fr/competitions/14264).

```shell
# Preprocess the data
python -u preprocess_datasets.py

# Finetune model - results are saved in `./logs/[save_prefix]`
python -u finetune.py --save_prefix v1 --target processed_query_converted --max_epochs 30 --gpus 0,1,2,3 --save_best --batch_size 10 --learning_rate 5e-6

# Inference - results are saved in `./logs/[resume_prefix]/inference_heldout.json`
python -u inference.py --input data/DBLP-QuAD/dblp.heldout.500.questionsonly.json --resume_prefix v1 --device 0 --batch_size 12 --use_convert

# Postprocess - results are saved in `./logs/[resume_prefix]/postprocess_heldout.json`
python -u postprocess.py --resume_prefix v1

# Do querying - results are saved in `./logs/[resume_prefix]/answer.txt`
python -u do_query.py --resume_prefix v1
```

----

## Citation

```
@inproceedings{DBLP:conf/semweb/0003ZRRB23,
  author       = {Ruijie Wang and
                  Zhiruo Zhang and
                  Luca Rossetto and
                  Florian Ruosch and
                  Abraham Bernstein},
  editor       = {Debayan Banerjee and
                  Ricardo Usbeck and
                  Nandana Mihindukulasooriya and
                  Gunjan Singh and
                  Raghava Mutharaju and
                  Pavan Kapanipathi},
  title        = {NLQxform: {A} Language Model-based Question to {SPARQL} Transformer},
  booktitle    = {Joint Proceedings of Scholarly {QALD} 2023 and SemREC 2023 co-located
                  with 22nd International Semantic Web Conference {ISWC} 2023, Athens,
                  Greece, November 6-10, 2023},
  series       = {{CEUR} Workshop Proceedings},
  volume       = {3592},
  publisher    = {CEUR-WS.org},
  year         = {2023},
  url          = {https://ceur-ws.org/Vol-3592/paper2.pdf},
  timestamp    = {Tue, 02 Jan 2024 17:44:44 +0100},
  biburl       = {https://dblp.org/rec/conf/semweb/0003ZRRB23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```