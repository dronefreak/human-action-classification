# 🧠 Major Model Families in `timm` (PyTorch Image Models)

The [`timm`](https://github.com/huggingface/pytorch-image-models) library (now maintained by Hugging Face) provides a vast collection of state-of-the-art computer vision models with pretrained weights, standardized interfaces, and support for training/inference.

As of **2025** (`timm >= 1.0`), it includes **1,200+ model variants** across diverse architectures.

Below are the **9 major model families** supported in `timm`, along with representative examples.

---

## 1. EfficientNet & Variants

Scalable CNNs using compound scaling. Includes original, Lite (for edge devices), and V2 versions.

- `efficientnet_b0 – efficientnet_b8`
- `efficientnet_lite0 – efficientnet_lite4`
- `efficientnetv2_s`, `efficientnetv2_m`, `efficientnetv2_l`, `efficientnetv2_xl`
- `tf_efficientnet_b0 – tf_efficientnet_l2` _(TensorFlow-trained weights)_

✅ **Ideal for mobile-to-server deployment** with excellent accuracy/efficiency trade-offs.

---

## 2. Vision Transformers (ViT) & Hybrids

Pure and hybrid transformer-based architectures for image recognition.

- **ViT:** `vit_tiny_patch16_224`, `vit_small_patch16_224`, `vit_base_patch16_224`, `vit_large_patch16_224`
- **DeiT (Data-efficient):** `deit_tiny_distilled_patch16_224`, `deit_base_distilled_patch16_384`
- **Swin Transformer:** `swin_tiny_patch4_window7_224`, `swin_base_patch4_window7_224`
- **BEiT:** `beit_base_patch16_224`, `beit_large_patch16_224`
- **Hybrids:** `convit_tiny`, `convmixer_768_32`

⚡ **Leverages self-attention**; excels with large datasets and high-resolution inputs.

---

## 3. ConvNeXt

Modern CNNs re-designed to match ViT performance using pure convolutional blocks.

- `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `convnext_xlarge`
- `convnextv2_tiny`, `convnextv2_base`, `convnextv2_large` _(with Global Response Normalization)_

🏆 **Combines CNN simplicity with ViT-level accuracy** — great for general-purpose use.

---

## 4. ResNet & ResNeXt Family

Classic and extended residual networks—workhorses of computer vision.

- **ResNet:** `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **ResNeXt:** `resnext50_32x4d`, `resnext101_32x8d`, `resnext101_64x4d`
- **Variants:** `seresnet50`, `wide_resnet50_2`, `regnetx_002`, `regnety_002`

🔧 **Reliable, well-understood**, and widely used in production systems.

---

## 5. MobileNets

Lightweight CNNs optimized for mobile and edge devices.

- `mobilenetv2_100`
- `mobilenetv3_small_100`, `mobilenetv3_large_100`
- `mobilenetv4_conv_small`, `mobilenetv4_conv_large` _(2024 release)_

📱 **Designed for low-latency, low-power inference** on phones and embedded hardware.

---

## 6. DenseNet, VGG, Inception

Foundational CNN architectures (still available for compatibility and research).

- **DenseNet:** `densenet121`, `densenet169`, `densenet201`
- **VGG:** `vgg11`, `vgg16`, `vgg19`
- **Inception:** `inception_v3`, `inception_v4`, `inception_resnet_v2`

📚 **Historically important**; less common in new deployments but useful for baselines.

---

## 7. Modern CNNs

Advanced CNN designs beyond ResNet, often combining attention, normalization, or novel blocks.

- **NFNet:** `nfnet_f0 – nfnet_f6`, `eca_nfnet_l0`
- **DLA:** `dla34`, `dla60`, `dla102`
- **DarkNet:** `darknet53`, `cspdarknet53`
- **CoAtNet:** `coatnet_0_224`, `coatnet_1_224`
- **MaxVit:** `maxxvit_rmlp_tiny`

🚀 **Push the limits of CNN performance** with modern techniques like _Squeeze-and-Excitation_, _ECA_, or _Co-Scale attention_.

---

## 8. MLPMixer & gMLP

Non-CNN, non-Transformer architectures based on multi-layer perceptrons.

- **MLP-Mixer:** `mixer_b16_224`, `mixer_l16_224`
- **gMLP:** `gmlp_b16_224`, `gmlp_ti16_224`

🌀 **Uses spatial and channel mixing via MLPs** — an alternative to attention-based models.

---

## 9. Miscellaneous & SOTA Models

Cutting-edge or specialized architectures from recent research.

- **XCiT:** `xcit_tiny_12_p16_224`, `xcit_small_12_p16_224` _(Cross-Covariance Image Transformers)_
- **PVT:** `pvt_v2_b0 – pvt_v2_b5` _(Pyramid Vision Transformer)_
- **PoolFormer:** `poolformer_s12`, `poolformer_m36`
- **LeViT:** `levit_128`, `levit_256`, `levit_384` _(CNN + ViT hybrid for fast inference)_
- **Others:** FocalNet, Davit, Hornet, etc.

🔮 **Represents the frontier of vision model research** — often high-performing but less standardized.

---
