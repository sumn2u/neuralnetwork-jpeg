# Relevant literature overview

* **Deep Convolution Networks for Compression Artifacts Reduction**
*Dong et al. April 2015 [link](https://arxiv.org/pdf/1504.06993.pdf)*
AR-CNN. Really the first paper about using CNNs for artifact reduction, often used as baseline in the other papers.

* **Compression Artifacts Removal Using Convolutional Neural Networks**
*Svoboda et al. 2016 [link](https://arxiv.org/pdf/1605.00366.pdf)*
Residual learning/skip architecture deep CNN, pretty simple (8 layers). Beats AR-CNN

* **Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising**
*Zhang et al. August 2016 [link](https://arxiv.org/pdf/1608.03981v1.pdf)*
Skip connection architecture, but as an added gimmick takes the residual image as the target.

* **CAS-CNN: A Deep Convolutional Neural Network for Image Compression Artifact Suppression**
*Cavigelli et al. November 2016 [link](https://arxiv.org/pdf/1611.07233v1.pdf)*
Deep, skip architecture with multiple exit points (loss at different depths). Similar to Unet (but much more complex architecture). The same architecture can do many tasks such as denoising, JPEG artifact removal, and more (large capacity).


* **D3: Deep Dual-Domain Based Fast Restoration of JPEG-Compressed Images**
*Zhang et al. 2014  [link](https://arxiv.org/pdf/1601.04149v3.pdf)*
End to end training of *One-Step Sparse Inference* modules which are efficient feed-forward approximations of sparse codings. Much faster and outperforms AR-CNN (like they all do).



* **Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections**
*Xiao-Jiao Mao et al. June 2016 [link](https://arxiv.org/pdf/1606.08921v3.pdf)*

* **One-to-Many Network for Visually Pleasing Compression Artifacts Reduction**
*Guo et al. November 2016 [link](https://arxiv.org/pdf/1611.04994v1.pdf)*
Approach that uses perceptual loss as a measure for training as well. Also, JPEG loss (if a pixel is out of possible bounds extra loss is applied).
They also do deconvolutions (learned upsampling) by what they call "shift-and-average", which I think is the same as shift-and-stitch described by Long et al. (original fully convolutional neural networks paper). This would eliminate grid-like artifacts due to upsampling and demonstrate a *"dramatic visual improvement"*.

* **FractalNet: Ultra-Deep Neural Networks without Residuals**
*Larsson et al. May 2016 [link](https://arxiv.org/pdf/1605.07648v2.pdf)*


* **U-Net: Convolutional Networks for Biomedical Image Segmentation**
*Ronneberger et al. May 2015 [link](https://arxiv.org/pdf/1505.04597v1.pdf)
Unet architecture for dense predictions, which is really per pixel regression.

* **Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising**
*Zhang et al. July 2017 [link](https://ieeexplore.ieee.org/document/7839189)*
A network to reduce Gaussian noise that does not require the level of noise.

* **End-to-end Optimized Image Compression**
*Ballé et al. March 2017 [link](https://arxiv.org/abs/1611.01704)*
Use of a nonlinear analysis transformation, a uniform quantizer, and a nonlinear synthesis transformation to optimize image compression.

* **BlockCNN: A Deep Network for Artifact Removal and Image Compression**
*Maleki et al. May 2018 [link](https://arxiv.org/abs/1805.11091)*
Use of BlockCNN that performs both artifact removal and image compression. Found good to low compression factors that don't predict high frequency details.

* **Learning a Single Model With a Wide Range of Quality Factors for JPEG Image Artifacts Removal**
*Li et al.  Sept 2020 [link](https://ieeexplore.ieee.org/document/9186829)*
The use of quantization tables as part of the training data  makes the proposed network a success.

* **Convolutional Neural Network for Image Compression with Application to JPEG Standard**
*Puchala & Stokfiszewski. March 2021 [link](https://ieeexplore.ieee.org/document/9418646)*
They use high-quality human face images to train the CNN model and compared it with other methods such as discrete cosine transform, lobed orthogonal transform, modulated lobed transform, and Karhunen-Loeve transform.

* **Primary Quantization Matrix Estimation of Double Compressed JPEG Images via CNN**
*Niu et al. December 2019 [link](https://ieeexplore.ieee.org/document/8945385)*
Use of dense CNN structure in fast forward fashion where early layers are directly used by deep layers throughout the same dense block.



