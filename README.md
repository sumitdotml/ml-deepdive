# ML Deepdive: Project-Based ML Learning

Currently prioritizing deep learning architecture design, optimization, and ML systems. Planning to also cover RL in the near future.

> [!NOTE]
> This is just a rough idea for now. I may end up modifying compartments of these pipelines at any point depending on my pace of learning, focus, and so on But for now, this is what I'm thinking.
>
> Basically, the only thing that is certain from here is the barebones network project. The others... not so certain. Just like how life goes.

There are 4 core pipelines here.

---

## Pipeline 1: Neural Network Fundamentals & Optimization

**Theme**: Math-driven NN design, PyTorch fluency, and optimization intuition.

### Projects:

#### 1. Autograd engine

[Project repo](https://github.com/sumitdotml/autograd) _*(in progress)_

#### 2. The Barebones Network

- **Task**: Implement a fully connected network from scratch in PyTorch (no nn.Module).

- **Focus**:
    - Raw tensor operations + autograd.
    - Derive backpropagation for a 2-layer network on paper.
    - Compare performance with/without batch normalization.

- **Math Layer**: Derive gradients for cross-entropy loss.

#

#### 3. Optimizer Olympics

- **Task**: Train the same model (e.g., ResNet-18 on CIFAR-10) with SGD, Adam, RMSProp, and a custom optimizer.

- **Focus**
    - Implement optimizers from scratch (override PyTorch’s torch.optim).
    - Visualize loss landscapes and convergence speed.

- **Math Layer**: Derive update rules for Adam (adaptive moments).

#

#### 4. Architecture Surgery

- **Task**: Modify a CNN (e.g., VGG) by:
    - Adding/removing skip connections (ResNet-style).
    - Swapping activation functions (Swish vs. ReLU).
    - Pruning 20% of weights and fine-tuning.

- **Goal**: Build intuition for how design choices affect performance.

---

## Pipeline 2: Advanced Architecture Design (Papers → Code)

**Theme**: Implement influential papers and experiment with variations.

### Projects:

#### 1. Build a Transformer from Scratch

- **Task**: Code a transformer for sequence-to-sequence tasks (e.g., time-series prediction).

- **Focus**:
    - Implement multi-head attention, positional encoding.
    - Compare with an LSTM baseline.

- **Paper Read**: Attention Is All You Need.

#

#### 2. Design a Hybrid Model

- **Task**: Combine CNNs and transformers (e.g., Vision Transformer) for image classification.

- **Stretch Goal**: Use PyTorch’s JIT to trace/script the model.

#

#### 3. Reproduce a Paper

- **Task**: Pick a recent arXiv paper (e.g., MLP-Mixer) and replicate its results.

- **Focus**: Debug discrepancies between paper and your implementation.

---

## Pipeline 3: ML Systems & Productionization

**Theme**: Deploy, optimize, and monitor models in real-world scenarios.

### Projects:

#### 1. Model Compression Challenge

- **Task**: Take a large model (e.g., BERT) and:
    - Prune 50% of its weights.
    - Quantize it to INT8 with PyTorch’s quantization tools.
    - Benchmark latency/accuracy on CPU vs. GPU.

#

#### 2. Deploy a Real-Time Inference API

- **Tools**: FastAPI, Docker, ONNX Runtime

- **Steps**:
    - Export a PyTorch model to ONNX.
    - Build a server with request batching and logging.
    - Stress-test with locust (simulate 1000+ RPS).

#

#### 3. Monitoring Pipeline

- **Task**: Track model drift in production:
    - Log predictions/confidence scores.
    - Detect data drift using statistical tests (KS test).

- **Tools**: Prometheus, Grafana, Python

---

## Pipeline 4: Capstone Project – End-to-End ML System

**Theme**: Combine everything into a single deployable product.

### Project:

- **Task**: Build a recommendation system with:
    - A custom neural architecture (e.g., graph neural nets for social data).
    - Distributed training (PyTorch + DDP).
    - A/B testing framework (compare with a matrix factorization baseline).
    - Deployment via TensorRT for low-latency inference.

- **Stretch Goal**: Optimize with C++ (e.g., write critical inference code in LibTorch).
