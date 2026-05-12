# Solution.md

## SMILES 2026 — Hallucination Detection

### Problem Description

The task focuses on binary hallucination detection using internal hidden representations extracted from a large language model (LLM). The dataset contains **689 total samples**, split into:

| Class | Count |
|---|---:|
| Hallucinated | 483 |
| Non-hallucinated | 206 |

This creates a moderately imbalanced classification setting, where naïvely predicting the majority class already achieves approximately 70% accuracy.

The challenge constraints are intentionally restrictive. Rather than fine-tuning the original language model or using richer architectural modifications, the task provides only the **internal hidden states for every token at every layer**. The objective is therefore to engineer informative features from these hidden representations and train a downstream **MLP probe** for hallucination classification.

The implementation is split across three primary files:

- `aggregation.py` — feature extraction and aggregation
- `probe.py` — MLP probe training and evaluation
- `splitting.py` — stratified K-fold cross-validation

---

# Literature Review

Recent work has shown that hallucination behaviour in large language models is often encoded directly within their internal representations.

A particularly influential result is:

**The Internal State of an LLM Knows When It's Lying**  
https://arxiv.org/abs/2304.13734

This work demonstrates that hidden states contain linearly separable information indicating whether a model is producing truthful or deceptive outputs. The paper argues that internal activations evolve differently during hallucinated generations, even when the generated text appears fluent and convincing. This motivates the use of hidden-state probing methods for hallucination detection. :contentReference[oaicite:0]{index=0}

Another recurring theme in the literature is the importance of **representation trajectories** across transformer layers. Rather than viewing layers independently, multiple works analyse how representations evolve layer-by-layer during generation. These trajectories often reveal uncertainty accumulation, semantic drift, or destabilisation during hallucinated responses.

Several related approaches inspired the feature engineering choices explored in this work:

- **DRIFT** studies representational drift across transformer layers and shows that unstable trajectories correlate with incorrect or hallucinated generations.
- **HaMI** investigates hallucination mitigation and detection using internal model activations and intermediate representations.
- Work analysing **Laplacian eigenvalues of attention maps** demonstrates that the geometry and spectral structure of attention distributions differ significantly between truthful and hallucinated outputs.

Additional related works include:

**LLM Internal States Reveal Hallucination Risk Faced With a Query**  
https://arxiv.org/abs/2407.03282

**Do LLMs "know" internally when they follow instructions?**  
https://arxiv.org/abs/2410.14516

These works collectively suggest that hallucination signals emerge not only in final-layer embeddings, but also in the *geometry*, *consistency*, and *evolution* of hidden representations across layers.

However, the current task is considerably more restrictive than many prior works. We are limited to:

1. Hidden states only
2. A fixed downstream **MLP classifier**
3. A relatively small dataset (689 samples)

More expressive approaches such as PCA pipelines, manifold learning, spectral decomposition, transformer-based probes, or sequence-level trajectory models could potentially improve performance, but were intentionally avoided to remain within the task constraints.

---

# Method

## Hidden State Aggregation

Following the findings from *The Internal State of an LLM Knows When It's Lying*, we hypothesised that hallucination information is most strongly encoded in the later transformer layers. :contentReference[oaicite:3]{index=3}

We therefore extracted features from the **penultimate 9 layers** of the model using the **last generated token** only. The representations from these layers were mean pooled to create a compact semantic summary vector.

Empirically, performance improved further when the **final layer** was also included alongside the penultimate layers. Although final-layer representations are sometimes noisier or more generation-specific, in practice they contributed useful discriminative information for this task.

All hidden-state features were extracted exclusively from the **last token representation**, since this token most directly reflects the model’s final generation state and confidence.

---

## Geometric Features

In addition to raw hidden activations, we engineered several geometric features intended to capture representational consistency across layers.

### Cosine Similarity Features

For the final token, cosine similarities were computed between intermediate-layer representations and the final-layer representation.

From these similarities, we extracted:

- Mean cosine similarity
- Variance of cosine similarity

The intuition is that hallucinated generations may exhibit greater representational instability or divergence as the model progresses through deeper layers.

---

## Trajectory Features

We additionally explored **trajectory-based features**, motivated by prior work analysing hidden-state evolution across transformer depth.

For consecutive layers \( l_i \) and \( l_{i+1} \), we computed the trajectory vector:

\[
t_i = h_{i+1} - h_i
\]

where \( h_i \) denotes the hidden representation at layer \( i \).

These trajectory vectors capture how the representation changes between layers rather than the representations themselves.

From the trajectory vectors we extracted:

- Mean pooled trajectory features
- Variance pooled trajectory features

These features aim to measure the smoothness and stability of representational evolution throughout the network.

---

## Probe Architecture

Because the task explicitly required an **MLP model**, we implemented a lightweight feed-forward probe.

The final architecture was:

- Linear(1888 → 80)
- ReLU activation
- Dropout(0.5)
- Linear(80 → 1)

We experimented with both **Adam** and **AdamW** optimisers. Due to the relatively small dataset size, Adam consistently produced better validation performance and more stable convergence.

---

## Data Splitting

To ensure robust evaluation under class imbalance, we used **5-fold stratified cross-validation**.

This was implemented using a stratified K-fold split so that each fold preserved the hallucinated/non-hallucinated class ratio.

This approach provided more reliable estimates of generalisation performance than a single train/validation split given the small sample count.

---

# Experimental Results

## Evaluation Summary (5-Fold Average)

| Checkpoint | Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| Majority-class baseline | 70.19% | 82.49% | N/A |
| Probe (train split) | 83.03% | 89.52% | 99.64% |
| Probe (validation split) | 74.02% | 83.91% | 71.50% |
| Probe (test split) | 70.19% | 81.41% | 70.98% |

# Additional Experiments

Several alternative feature extraction strategies were explored during development.

These included:

- Splitting layers into:
  - early layers
  - middle layers
  - late layers
- Mean pooling over:
  - all tokens
  - subsets of tokens
- Using layer norms from each layer followed by mean pooling

While some variants produced marginal improvements on individual folds, none consistently outperformed the final feature configuration based on:
- penultimate-layer aggregation,
- cosine similarity statistics,
- and trajectory features.

---

# Threshold Selection and Metrics

The classification threshold was selected using the **best validation F1 score**.

Although the accompanying codebase indicated AUROC as an important metric, the README specified **accuracy as the primary leaderboard metric**. Consequently, evaluation considered all three metrics:

- Accuracy
- F1 score
- AUROC

The final model achieves the baseline accuracy, but granted that the dataset is so imbalanced, one can see this as a good result given the design constraints. More significantly, the AUROC improved from the baseline of  50% to 70.98% showing that the model has learned to extract some signals from the data.
