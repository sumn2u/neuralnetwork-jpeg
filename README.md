# Convolutional Neural Network (CNN) Image Compression

Encode/decode natural images using Neural Network.


**Motivation :**
> That said, a research paper must include some original contributions,
introducing novel ideas and proving its validity and improved performance.
In fact, writing a research paper is the culmination of an intense dedication to research,
having analysed different hypothesis and assessed their appropriateness.

**Goals :**
- use different image compression techniques to get compressed images
- compare different image enhancement algorithms
- purposed own algorithm
- compare it to other alogorithms 

## Status

Work in progress
--updated by Suman kunwar

## Try it out

```bash
$ cd npeg
$ ipython -i train.py
```

then type in one of the following commands:

```python
# strong noise training (should converge to 0.0008 loss)
for i in range(5):
    r(cnoise=15.0)

# weak noise pretraining (no difference in final loss)
for i in [0.1, 0.3, 1.0, 5.0, 15.0]:
    r(cnoise=i)

# save weights to file
save()

# load weights from file
load()

# test model on randomly sampled CIFAR
show()
```

Dependencies:

- TensorFlow r1.0
- Canton library: `pip install canton`
- Keras (mainly for downloading CIFAR)

If you need visualization functionality:

- cv2 (install via conda is recommended)
- cv2tools (pull from <https://github.com/ctmakro/cv2tools> then `pip install -e <dirname>`)
- or modify the code to use other visualization libraries you prefer

## How it works

> CNN Image Compression - Neural Network Image Compression：<https://sumn2u.github.io/neuralnetwork-jpeg>

Training:

- image -> Encoder CNN -> features
- features += gaussian noise
- features -> sigmoid -> code
- code -> Decoder CNN -> reconstruction
- loss = mean((image-reconstruction) ** 2) + mean(code**2) * 0.01

To reduce reconstruction loss, the best encoding strategy for the encoder is to drive its output ("features") large, to reduce artifacts caused by the gaussian noise.

Therefore by increasing the magnitude of the gaussian noise, the code will eventually saturate to 0 or 1.

We encourage sparsity of the code (to allow for further compression) by adding a penalty term (`mean(code**2) * 0.01`), after which the code will tend to include more zeros and less ones.

Testing:

- image -> Encoder CNN -> features
- features -> sigmoid -> binary quantization -> code
- code -> Decoder CNN -> reconstruction

## About

Original Author: Qin Yongliang and Suman Kunwar


License: MIT
