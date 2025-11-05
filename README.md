


# LISA: A Layer-wise Integration and Suppression Approach for Hallucination Mitigation in Multimodal Large Language Models

This repository contains the official implementation of **LISA**, a method designed to mitigate hallucinations in **Multimodal Large Language Models (MLLMs)** through layer-wise integration and suppression. **LISA** leverages spectral regularization and token-level fusion across transformer layers to enhance model stability and grounding.

> **Paper**: *Coming soon*
> **Compatible Models**: LLaVA, InstructBLIP, Qwen-VL, Qwen2.5-VL
> **Evaluated on**: POPE, CHAIR, MME, AMBER



## Installation

To get started with **LISA**, follow these steps:

```bash
cd LISA
conda create -n lisa_env python=3.10
conda activate lisa_env
pip install -r requirements.txt
```






## Key Features

* Compatible with major MLLM backbones (LLaVA, InstructBLIP, Qwen-VL, Qwen2.5-VL).
* Supports various decoding strategies: greedy, beam search, and nucleus sampling.
* Tools for spectral trace logging and visualization.
* Configurable anchor routing and fusion policies.
* Evaluation scripts for benchmark datasets: POPE, CHAIR, MME, and AMBER.








