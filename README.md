# LISA: A Layer-wise Integration and Suppression Approach for Hallucination Mitigation in Multimodal Large Language Models

This repository contains the official implementation of **LISA**, a method designed to mitigate hallucinations in Multimodal Large Language Models (MLLMs) by leveraging spectral regularization and token-level fusion across transformer layers.

>  Paper: *Coming soon*  
>  Compatible Models: LLaVA, InstructBLIP, Qwen-VL, Qwen2.5-VL  
>  Evaluated on: POPE, CHAIR, MME, AMBER

---

##  Motivation

Multimodal models often exhibit hallucinations, especially in the deep layers of transformer decoders. **LISA** introduces a layer-wise perspective to address this issue through:

- **Spectral Modulation**: Suppressing unstable deep-layer activations while preserving early-layer grounding;
- **Cross-layer Fusion**: Aggregating token-level logits across selected anchor layers;
- **Token-wise Routing**: Dynamically selecting reliable layers for each token.

---

##  Key Features

-  Compatible with major MLLM backbones (LLaVA, InstructBLIP, Qwen-VL, Qwen2.5-VL)  
-  Support for multiple decoding strategies: greedy, beam, nucleus  
-  Spectral trace logging and visualization tools  
-  Configurable anchor routing and fusion policies  
-  Evaluation scripts for POPE, CHAIR, MME, AMBER benchmarks

---

##  Installation

```bash

cd LISA
conda create -n lisa_env python=3.10
conda activate lisa_env
pip install -r requirements.txt
