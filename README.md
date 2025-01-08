# LDM-Morph

This repository is the official implementation for the paper:  
**[LDM-Morph: Latent diffusion model guided deformable image registration](https://arxiv.org/pdf/2411.15426)**  
*Authors: Jiong Wu and Kuang Gong*

>Abstract: Deformable image registration plays an essential role in various medical image tasks. Existing deep learning-based deformable registration frameworks primarily utilize convolutional neural networks (CNNs) or Transformers to learn features to predict the deformations. However, the lack of semantic information in the learned features limits the registration performance. Furthermore, the similarity metric of the loss function is often evaluated only in the pixel space, which ignores the matching of high-level anatomical features and can lead to deformation folding. To address these issues, in this work, we proposed LDM-Morph, an unsupervised deformable registration algorithm for medical image registration. LDM-Morph integrated features extracted from the latent diffusion model (LDM) to enrich the semantic information. Additionally, a latent and global feature-based cross-attention module (LGCA) was designed to enhance the interaction of semantic information from LDM and global information from multihead self-attention operations. Finally, a hierarchical metric was proposed to evaluate the similarity of image pairs in both the original pixel space and latent feature space, enhancing topology preservation while improving registration accuracy. Extensive experiments on four public 2D cardiac image datasets show that the proposed LDM-Morph framework outperformed existing state-of-the-art CNNs- and Transformers-based registration methods regarding accuracy and topology preservation with comparable computational efficiency.



---

## Overview

### Figure Overview
Below is an example of the core results or architecture from our paper:

<p align="center">
  <img src="path/to/figure1.png" alt="Figure 1 Overview" width="600">
  <br>
  <em>Figure 1. A description of the figure.</em>
</p>

---

## Features

- Briefly list the key features or objectives of your project.
  - Feature 1
  - Feature 2
  - Feature 3

---

## Installation

### Requirements
- Python >= 3.x
- Libraries: `torch`, `numpy`, etc.

Install dependencies:
```bash
pip install -r requirements.txt
