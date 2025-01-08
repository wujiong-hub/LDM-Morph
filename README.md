# LDM-Morph

This repository is the official implementation for the paper:  
**[LDM-Morph: Latent diffusion model guided deformable image registration](https://arxiv.org/pdf/2411.15426)**  
*Authors: Jiong Wu and Kuang Gong*

>Abstract: Deformable image registration plays an essential role in various medical image tasks. Existing deep learning-based deformable registration frameworks primarily utilize convolutional neural networks (CNNs) or Transformers to learn features to predict the deformations. However, the lack of semantic information in the learned features limits the registration performance. Furthermore, the similarity metric of the loss function is often evaluated only in the pixel space, which ignores the matching of high-level anatomical features and can lead to deformation folding. To address these issues, in this work, we proposed LDM-Morph, an unsupervised deformable registration algorithm for medical image registration. LDM-Morph integrated features extracted from the latent diffusion model (LDM) to enrich the semantic information. Additionally, a latent and global feature-based cross-attention module (LGCA) was designed to enhance the interaction of semantic information from LDM and global information from multihead self-attention operations. Finally, a hierarchical metric was proposed to evaluate the similarity of image pairs in both the original pixel space and latent feature space, enhancing topology preservation while improving registration accuracy. Extensive experiments on four public 2D cardiac image datasets show that the proposed LDM-Morph framework outperformed existing state-of-the-art CNNs- and Transformers-based registration methods regarding accuracy and topology preservation with comparable computational efficiency.



---

## Overview

<p align="center">
  <img src="documents/fig1_architecture.jpg" alt="Figure 1 Overview" width="800">
  <br>
</p>

---

## Environment

conda create -n universalmodel python=3.7
conda activate universalmodel
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 
## please modify according to the CUDA version in your server
pip install 'monai[all]'
pip install -r requirements.txt
cd pretrained_weights/
wget https://huggingface.co/ljwztc/CLIP-Driven-Universal-Model/resolve/main/clip_driven_universal_swin_unetr.pth?download=true
cd ../
python pred_pseudo.py --data_root_path PATH_TO_IMG_DIR --result_save_path PATH_TO_result_DIR --resume ./pretrained_weights/clip_driven_universal_swin_unetr.pth
## For example: python pred_pseudo.py --data_root_path /home/data/ct/ --result_save_path /home/data/result --resume ./pretrained_weights/clip_driven_universal_swin_unetr.pth
