# NPEG - Neural Network Image Compression

Training:

- image -> Encoder CNN -> features
- features += gaussian noise
- features -> sigmoid -> code
- code -> Decoder CNN -> reconstruction
- loss = mean((image-reconstruction) ** 2) + mean(code**2) * 0.01
