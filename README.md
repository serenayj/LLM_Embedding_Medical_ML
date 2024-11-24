# LLM Embedding for Medical Machine Learning Applications

This repository contains the source code and supplementary materials for our EMNLP Findings 2024 paper:

**[When Raw Data Prevails: Are Large Language Model Embeddings Effective in Numerical Data Representation for Medical Machine Learning Applications?](https://aclanthology.org/2024.findings-emnlp.311/)**

## Overview

The paper investigates the effectiveness of large language model (LLM) embeddings in representing numerical data from electronic health records (EHRs) for medical diagnostics and prognostics. We evaluate the utility of LLM-derived embeddings as features for machine learning classifiers in tasks like diagnosis prediction, mortality prediction, and length-of-stay prediction. Our findings highlight:

- **Raw data features dominate** medical ML tasks, but LLM embeddings demonstrate competitive performance in some cases.
- LLM embeddings are robust across tasks and settings, providing an alternative to raw data for certain applications.
- This work emphasizes the potential of LLMs for medical applications and identifies challenges in their current use.

## Features

- **Table-to-Text Conversion:** Scripts to transform tabular EHR data into formats like narratives and JSON for LLM input.
- **Embedding Classifiers:** Pipelines to train and evaluate ML classifiers using embeddings extracted from the last hidden states of LLMs.
- **XGB Baseline:** Implementation of XGBoost models using raw numerical EHR data as input for baseline comparison.

## Repository Structure

- `table_to_text_conversion.py`: Converts tabular EHR data into narrative or structured text formats.
- `embedding_classifiers.py`: Extracts embeddings from LLMs and trains classifiers.
- `xgb_baseline.py`: Implements the XGBoost baseline using raw numerical features.


## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required libraries (install via `requirements.txt`):
  ```bash
  pip install -r requirements.txt
 
