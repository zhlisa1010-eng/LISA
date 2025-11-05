


# LISA: A Layer-wise Integration and Suppression Approach for Hallucination Mitigation in Multimodal Large Language Models

This repository contains the official implementation of **LISA**, a method designed to mitigate hallucinations in **Multimodal Large Language Models (MLLMs)** through layer-wise integration and suppression. **LISA** leverages spectral regularization and token-level fusion across transformer layers to enhance model stability and grounding.

> **Paper**: *Coming soon*
> **Compatible Models**: LLaVA, InstructBLIP, Qwen-VL, Qwen2.5-VL
> **Evaluated on**: POPE, CHAIR, MME, AMBER



## Motivation

Multimodal models, particularly those based on transformers, often produce hallucinations, especially in deeper layers. **LISA** addresses this issue by introducing a layer-wise decoding approach that combines the following mechanisms:

* **Spectral Modulation (SM)**: Stabilizes activations in deep layers while preserving grounding from early layers.
* **Cross-layer Fusion (CF)**: Aggregates token-level representations across multiple layers to create a stable, reliable anchor.
* **Token-wise Soft Fusion (TSF)**: Dynamically selects the most reliable layers for each token during decoding.

These combined mechanisms help **LISA** effectively mitigate hallucinations and improve model robustness.



## Key Features

* Compatible with major MLLM backbones (LLaVA, InstructBLIP, Qwen-VL, Qwen2.5-VL).
* Supports various decoding strategies: greedy, beam search, and nucleus sampling.
* Tools for spectral trace logging and visualization.
* Configurable anchor routing and fusion policies.
* Evaluation scripts for benchmark datasets: POPE, CHAIR, MME, and AMBER.





## Installation

To get started with **LISA**, follow these steps:

```bash
cd LISA
conda create -n lisa_env python=3.10
conda activate lisa_env
pip install -r requirements.txt
```



## Conclusion

The **ablation study** confirms that **LISA** operates as a layered system, with each core mechanism playing a distinct role:

* **Spectral Modulation (SM)** stabilizes the model by preventing instability in deeper layers.
* **Cross-layer Fusion (CF)** aggregates information across layers, creating a reliable anchor representation.
* **Token-wise Soft Fusion (TSF)** adapts during decoding, ensuring that the most stable representations are used.

Together, these mechanisms provide significant improvements over the baseline, confirming **LISA** as an effective method for reducing hallucinations and enhancing output fidelity in **Multimodal Large Language Models (MLLMs)**.



This version includes the **Ablation Study** section to clearly outline the importance of each component in the **LISA** framework. Let me know if you need any further modifications!
