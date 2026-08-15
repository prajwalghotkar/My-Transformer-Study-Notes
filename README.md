# My-Transformer-Study-Notes

----

#  Attention Is All You Need — My Transformer Study Notes
 
> Deep-dive notes on the paper that changed NLP forever: **Vaswani et al., 2017 — "Attention Is All You Need"**
> Written while I was learning Transformers from scratch, with a mini project built alongside these notes.
 
![Status](https://img.shields.io/badge/status-learning-yellow)
![Topic](https://img.shields.io/badge/topic-Transformers-blue)
![Paper](https://img.shields.io/badge/paper-arXiv%3A1706.03762-red)
 
---
 
##  Table of Contents
 
1. [Why This Paper Matters](#-why-this-paper-matters)
2. [The Core Idea](#-the-core-idea)
3. [Problem With RNNs/CNNs (Before Transformers)](#-problem-with-rnnscnns-before-transformers)
4. [Overall Architecture](#-overall-architecture)
5. [Encoder Stack](#-encoder-stack)
6. [Decoder Stack](#-decoder-stack)
7. [Attention Mechanism](#-attention-mechanism)
   - [Scaled Dot-Product Attention](#scaled-dot-product-attention)
   - [Multi-Head Attention](#multi-head-attention)
   - [Where Attention Is Used](#where-attention-is-used-in-the-model)
8. [Position-wise Feed-Forward Networks](#-position-wise-feed-forward-networks)
9. [Embeddings & Softmax](#-embeddings--softmax)
10. [Positional Encoding](#-positional-encoding)
11. [Why Self-Attention? (Complexity Comparison)](#-why-self-attention-complexity-comparison)
12. [Training Details](#-training-details)
13. [Results](#-results)
14. [Key Takeaways](#-key-takeaways)
15. [My Project](#-my-project)
16. [Resources I Used](#-resources-i-used)
---
 
##  Why This Paper Matters https://arxiv.org/pdf/1706.03762
 
Before this paper (2017), almost every good NLP model was built on **RNNs / LSTMs / GRUs**, optionally combined with attention. This paper asked a bold question:
 
> *"What if we throw away recurrence and convolutions completely, and build a model using **only attention**?"*
 
The result was the **Transformer** — the architecture behind BERT, GPT, T5, and basically every modern LLM .
 
---
 
##  The Core Idea
 
- Ditch recurrence (no more processing word-by-word in sequence).
- Ditch convolutions (no more sliding windows).
- Use **self-attention** to let every word directly "look at" every other word in the sentence, no matter how far apart they are.
- This makes the model **highly parallelizable** → much faster to train on GPUs.
---
 
##  Problem With RNNs/CNNs (Before Transformers)
 
| Model Type | Problem |
|---|---|
| **RNN / LSTM / GRU** | Sequential by nature — word *t* depends on word *t-1*. Can't parallelize within a sentence → slow training. Also struggles with **long-range dependencies** (info from far-away words fades). |
| **CNN (ConvS2S, ByteNet)** | Can parallelize better, but to connect two far-apart words you need many stacked layers → path length grows with sequence length (linearly or logarithmically). |
| **Self-Attention (Transformer)** | Any two words are connected in **O(1)** steps — one attention operation, regardless of distance. Fully parallelizable. |
 
---
 
##  Overall Architecture
 
The Transformer follows the classic **Encoder–Decoder** structure, but built entirely from attention + feed-forward layers.
 
<img width="2720" height="2300" alt="transformer_architecture" src="https://github.com/user-attachments/assets/bf721fab-815a-4f14-96a4-81b9598f8e65" />


 
- Encoder: maps input tokens → continuous representations.
- Decoder: takes encoder output + previously generated tokens → predicts the next token, one at a time (**auto-regressive**).
---
 
## Encoder Stack
 
- Made of **N = 6 identical layers**.
- Each layer has **2 sub-layers**:
  1. **Multi-Head Self-Attention**
  2. **Position-wise Feed-Forward Network**
- Each sub-layer wrapped with a **residual connection + LayerNorm**:
```
output = LayerNorm(x + Sublayer(x))
```
 
- Every sub-layer (and embedding layer) outputs vectors of dimension `d_model = 512`, so residual addition works cleanly.
---
 
##  Decoder Stack
 
- Also **N = 6 identical layers**, but each layer has **3 sub-layers**:
  1. **Masked Multi-Head Self-Attention** — prevents a position from "seeing" future tokens (crucial so the model can't cheat during training).
  2. **Encoder-Decoder (Cross) Attention** — queries come from decoder, keys/values come from encoder output. This is how decoder "looks at" the input sentence.
  3. **Position-wise Feed-Forward Network**
- Same residual + LayerNorm pattern as encoder.
- **Masking trick**: set illegal (future) positions to `-∞` before softmax → their attention weight becomes 0.
---
 
##  Attention Mechanism
 
### Scaled Dot-Product Attention
 
The heart of the whole paper. Given:
- **Q** (queries), **K** (keys), **V** (values)
```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V
```
 
**Step by step:**
1. Compute similarity between query and every key → `Q · Kᵀ`
2. Scale down by `√d_k` (prevents huge dot products when `d_k` is large, which would push softmax into tiny-gradient regions)
3. Apply `softmax` → converts scores into attention weights (sum to 1)
4. Multiply weights with `V` → weighted sum = final output
**Why scale by √d_k?**
If `q` and `k` components are random with mean 0, variance 1, then `q·k` has variance `d_k`. Without scaling, large `d_k` → large dot products → softmax gradients vanish. Dividing by `√d_k` keeps things stable.
 
### Multi-Head Attention
 
Instead of doing attention once with full `d_model` dimensions, split into **h = 8 heads**, each working in a smaller `d_k = d_v = d_model/h = 64` dimensional space.
 
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_8) · W_O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```
 
**Why multiple heads?**
- A single attention head averages over everything — it can only focus on "one kind" of relationship at a time.
- Multiple heads let the model attend to **different types of relationships simultaneously** — e.g., one head might track syntax, another might resolve pronouns (anaphora), another might track long-distance dependencies. (See attention visualizations in the paper — heads genuinely specialize!)
- Total compute stays similar to single-head attention because each head works on a smaller dimension.
### Where Attention Is Used in the Model
 
| Location | Q comes from | K, V come from | Purpose |
|---|---|---|---|
| **Encoder self-attention** | previous encoder layer | previous encoder layer | Every input word attends to every other input word |
| **Decoder self-attention (masked)** | previous decoder layer | previous decoder layer (masked) | Every output word attends to *previous* output words only |
| **Encoder-Decoder attention** | decoder | encoder output | Every output position attends to the entire input sequence |
 
---
 
##  Position-wise Feed-Forward Networks
 
Applied identically (but independently) to every position:
 
```
FFN(x) = max(0, x·W1 + b1)·W2 + b2
```
 
- Two linear layers with a ReLU in between.
- Input/output dim = 512, inner (hidden) dim = 2048.
- Think of it as a "per-token" mini neural network — same weights used at every position, but parameters differ from layer to layer.
---
 
##  Embeddings & Softmax
 
- Input tokens and output tokens are converted to vectors via **learned embeddings** (dim = `d_model`).
- The **same weight matrix** is shared between: input embedding, output embedding, and the pre-softmax linear layer (weight tying — reduces parameters).
- Embedding weights are multiplied by `√d_model` before adding positional encoding.
---
 
##  Positional Encoding
 
Since there's **no recurrence and no convolution**, the model has no built-in sense of word order. So we inject position info manually:
 
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
 
- Added directly to input/output embeddings (same dimension, so they can be summed).
- Uses sine/cosine of varying frequencies → each dimension corresponds to a different wavelength (from `2π` to `10000·2π`).
- **Key benefit**: for any fixed offset `k`, `PE(pos+k)` can be expressed as a linear function of `PE(pos)` — this may help the model learn to attend by *relative* position.
- Learned positional embeddings were also tested → nearly identical performance, but sinusoidal version can potentially extrapolate to longer sequences than seen in training.
---
 
##  Why Self-Attention? (Complexity Comparison)
 
| Layer Type | Complexity per Layer | Sequential Ops | Max Path Length |
|---|---|---|---|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k(n)) |
| Self-Attention (restricted) | O(r·n·d) | O(1) | O(n/r) |
 
*(n = sequence length, d = model dimension, k = kernel size, r = neighborhood size)*
 
**Takeaway:** Self-attention connects any two positions in a constant number of operations — this is what makes long-range dependency learning much easier than RNNs, and it's more parallelizable than both RNNs and CNNs (when n < d, which is the common case for sentence-level tasks).
 
---
 
##  Training Details
 
- **Dataset**: WMT 2014 English-German (~4.5M sentence pairs, byte-pair encoding, ~37K shared vocab) and English-French (36M sentences, 32K word-piece vocab).
- **Hardware**: 8× NVIDIA P100 GPUs.
- **Base model**: 100,000 steps, ~12 hours.
- **Big model**: 300,000 steps, ~3.5 days.
- **Optimizer**: Adam, β1=0.9, β2=0.98, ε=1e-9, with a custom learning-rate schedule:
```
lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```
 
  → learning rate increases linearly for the first `warmup_steps` (4000), then decays proportional to inverse square root of step number.
 
- **Regularization**:
  - Residual Dropout (`P_drop = 0.1`) applied after each sub-layer and to embeddings + positional encodings.
  - Label Smoothing (`ε_ls = 0.1`) — hurts perplexity slightly but improves BLEU/accuracy.
---
 
##  Results
 
| Task | BLEU Score | Notable Detail |
|---|---|---|
| WMT'14 EN→DE | **28.4** | New SOTA, beating previous best ensembles by 2+ BLEU |
| WMT'14 EN→FR | **41.8** | New single-model SOTA, at a fraction of training cost |
| English Constituency Parsing | 91.3 (WSJ only) / 92.7 (semi-supervised) | Shows Transformer generalizes beyond translation |
 
**Key experiment insights (ablations, Table 3 of paper):**
- Too few or too many attention heads both hurt performance — there's a sweet spot.
- Reducing `d_k` (key dimension) hurts quality → dot-product compatibility needs enough dimensions.
- Bigger models = better (as expected), and dropout meaningfully helps prevent overfitting.
- Learned positional embeddings ≈ sinusoidal positional embeddings in performance.
---
 
##  Key Takeaways
 
-  Attention alone (no recurrence/convolution) can outperform RNN/CNN-based models.
-  Self-attention gives **O(1)** path length between any two tokens → excellent for long-range dependencies.
-  Multi-head attention lets the model learn multiple types of relationships in parallel.
-  Positional encoding is a clever hack to inject sequence order without recurrence.
-  Parallelization = faster training = the real unlock that made scaling LLMs possible.
-  This single architecture became the foundation for BERT, GPT, T5, Vision Transformers, and more.
---

 
##  Resources I Used
 
- 📄 Original Paper: *Attention Is All You Need* (Vaswani et al., 2017) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- 💻 Official code reference: [tensor2tensor (Google)](https://github.com/tensorflow/tensor2tensor)

---
 
 
