# Rough Project Idea Dump

## Core Architecture Components

Theme: Understand and implement fundamental building blocks of modern architectures.

### Projects:

#### 1. The Anatomy of Attention

Task: Implement multi-head self-attention from scratch in PyTorch.

Focus:

Derive the attention equation (Q, K, V) and its gradients.

Compare efficiency of manual CUDA kernels vs. PyTorch’s F.scaled_dot_product_attention.

Experiment: Replace attention with a fixed Fourier transform layer – how does performance change?


#### 2. Normalization Wars

Task: Train the same ResNet-18 on ImageNet-1k with:

BatchNorm

LayerNorm

InstanceNorm

GroupNorm

Math Layer: Derive the gradient updates for BatchNorm.

Analysis: Measure training stability and convergence speed for each variant.

#### 3. Dynamic Architectures

Task: Implement a Mixture-of-Experts (MoE) layer and integrate it into a transformer.

Paper Read: GShard: Scaling Giant Models with Conditional Computation.

Stretch Goal: Benchmark throughput vs. dense models on synthetic data.

## Foundation Model Architectures

Theme: Deconstruct, implement, and modify landmark architectures.

### Projects:

#### 1. CLIP-Style Contrastive Learning

Task: Train a dual-encoder model (image + text) on a small dataset (e.g., Flickr8k).

Focus:

Implement the contrastive loss (InfoNCE).

Probe zero-shot transfer (e.g., classify images using text prompts).

Paper Read: Learning Transferable Visual Models From Natural Language Supervision.

#### 2. Diffusion Model from Scratch

Task: Code a DDPM (Denoising Diffusion Probabilistic Model) for image generation.

Math Layer: Derive the variational lower bound (ELBO) for diffusion.

Stretch Goal: Implement acceleration techniques (DDIM, latent diffusion).

#### 3. Sparsely Activated Giants

Task: Replicate a scaled-down version of Switch Transformer (MoE + Transformers).

Focus:

Implement token routing logic.

Profile memory usage vs. dense models.

## Architecture Innovation Lab

Theme: Design novel variants and analyze their behavior.

### Projects:

#### 1. Architecture Search Lite

Task: Use Neural Architecture Search (NAS) with PyTorch + Optuna to find optimal CNN cells for CIFAR-10.

Constraint: Limit search to 10 GPU hours.

#### 2. Hybrid Architecture Challenge

Task: Combine two disparate paradigms (e.g., CNN + Transformer + Liquid Neural Networks) for video prediction.

Analysis: Ablate each component’s contribution to final performance.

#### 3. The Minimalist Model

Task: Design the smallest possible architecture (e.g., <1M params) that achieves >90% on MNIST.

Twist: Use only convolutional layers with kernel size 1x1 and no dense layers.

## Systems for Foundation Models

Theme: Engineering techniques to scale and deploy giant architectures.

### Projects:

#### 1. Distributed Training Bootcamp

Task:

Train a model using:

PyTorch’s DDP (Distributed Data Parallel)

Pipeline Parallelism with GPipe

Fully Sharded Data Parallelism (FSDP)

Metric: Compare throughput and max achievable batch size.

#### 2. Quantization-Aware Training (QAT)

Task: Take a Vision Transformer (ViT) and quantize it to 4-bit precision without catastrophic accuracy loss.

Tools: PyTorch torch.quantization, Brevitas.

#### 3. On-Device Inference

Task: Deploy a LLM (e.g., TinyLlama) on a Raspberry Pi using:

ONNX Runtime

Llama.cpp-style C++ bindings

Optimization: Kernel fusion, KV-cache optimization.
