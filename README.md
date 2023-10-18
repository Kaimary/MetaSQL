# <span style="font-size:0.8em;">M</span>etaSQL: A Generate-then-Rank Framework for Natural Language to SQL Translation
> Improve NL2SQL with a unified generate-and-rank pipeline
> 
The official repository which contains the code and pre-trained models for our paper [MetaSQL: A Generate-then-Rank Framework for Natural Language to SQL Translation](https://arxiv.org/) (Not published yet😊).

<p align="center">
   <a href="https://github.com/kaimary/GAR/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/kaimary/metaSQL.svg?color=blue">
   </a>
   <a href="https://github.com/kaimary/GAR/stargazers">
       <img alt="stars" src="https://img.shields.io/github/stars/kaimary/metaSQL" />
  	</a>
  	<a href="https://github.com/kaimary/GAR/network/members">
       <img alt="FORK" src="https://img.shields.io/github/forks/kaimary/metaSQL?color=FF8000" />
  	</a>
    <a href="https://github.com/kaimary/GAR/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/kaimary/metaSQLs?color=0088ff"/>
    </a>
    <br />
</p>

## Overview

This code implements:

* The MetaSQL method for NL2SQL task decomposition.
* A unified <strong>generate-and-rank</strong> pipeline to improve existing neural Seq2seq NL2SQL models and LLMs. 

### About MetaSQL
> **TL;DR:** We introduce MetaSQL -- a unified generate-and-rank pipeline that is compatible with any existing NL2SQL models to improve their translation accuracy.
> METASQL introduces query metadata to control the generation of better SQL query candidates and use learning-to-rank algorithms to retrieve globally optimized queries.

The objective of NL2SQL translation is to convert a natural language query into a SQL query. 

Although existing approaches have shown promising results on standard benchmarks, the single SQL queries generated by auto-regressive decoding may
result in sub-optimal outputs in two main aspects: (1) *Lack of output diversity*. Auto-regressive decoding, commonly used with beam search or sampling methods, such as top-k sampling, often struggles with generating a diverse set of candidate sequences and tends to exhibit repetitiveness in its outputs; (2) *Lack of global context awareness*. Due to the incremental nature of generating output tokens one by
one based on the previously generated tokens, auto-regressive decoding may lead to encountering local optima outputs as it considers only partial context, thereby causing a failure to find the correct translation as well.

To tackle the problem of insufficient output diversity, we introduce query metadata to manipulate the behavior of translation models for better SQL query candidate generation. Moreover, to address the lack of global context, we reframe the NL2SQL problem as a ranking task, effectively leveraging the entire global context rather than the partial information involved in sequence generation. 

This is the approach taken by the MetaSQL method.

### How it works

MetaSQL uses the following three steps to do the translation:

1. **Semantic Decomposition**: Decompose the meaning of the given NL query and map it to a set of query metadata.
2. **Metadata-conditioned Generation**: Manipulate the behaviour of the translation model to generate a collection of SQL queries by conditioning on different compositions of the retrieved metadata.
3. **Two-stage Ranking Pipeline**: Rank based on the semantic similarity with a given NL query and find the closest SQL query as the translation result.

This process is illustrated in the diagram below:

<div style="text-align: center">
<img src="assets/overview.png" width="800">
</div>


## Quick Start

### Prerequisites
First, you should set up a python environment. This code base has been tested under python 3.8.

1. Install the required packages
```bash
bash env.sh
```
2. Download the [Spider](https://yale-lily.github.io/spider) dataset, and put the data into the <strong>data</strong> folder. Unpack the dataset and create the following directory structure:
```
/data
├── database
│   └── ...
├── dev.json
├── dev_gold.sql
├── tables.json
├── train_gold.sql
├── train_others.json
└── train_spider.json
```

### Evaluation
The evaluation script is located in root directory `test_pipeline.sh`.
You can run it with:
```
$ bash test_pipeline.sh <test_file_path> <table_path> <db_dir>
```

The evaluation script will create the directory `output` in the current directory.
The evaluation results will be stored there.

## Note
The pretrained models will be uploaded after our paper is accepted. 😊
