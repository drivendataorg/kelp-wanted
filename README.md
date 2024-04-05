[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

<a href="https://www.drivendata.org/competitions/255/kelp-forest-segmentation/"><img alt="Overhead drone footage of giant kelp canopy. Image Credit: Tom Bell, All Rights Reserved." src="https://drivendata-public-assets.s3.amazonaws.com/kelp-canopy-2.jpg" style="
    object-fit: cover;
    object-position: center;
    height: 250px;
    width: 100%;
"></a>

# Kelp Wanted: Segmenting Kelp Forests

## Goal of the Competition

The goal of this challenge was to detect the presence of kelp canopy using Landsat satellite imagery. The labels were generated by citizen scientists participating in the [Floating Forests](https://www.zooniverse.org/projects/zooniverse/floating-forests) project. Competitors were tasked with performing binary semantic segmetnation on 350x350 tiles that depicted coastlines around the [Falkland Islands](https://en.wikipedia.org/wiki/Falkland_Islands). Kelp forests are essential marine ecosystems supporting diverse species and offering substantial economic value. Satellite imagery offers a cost-effective and scalable method to map and monitor these habitats, crucial for safeguarding these vital habitats against climate change, overfishing, and unsustainable activities.

## What's in this Repository

This repository contains code from winning competitors in the [Kelp Wanted: Segmenting Kelp Forests](https://www.drivendata.org/competitions/255/) DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Private Score | Public Score | Summary of Model
--- | --- | --- | --- | ---
1  | Team Epoch IV: [EWitting](https://www.drivendata.org/users/EWitting/), [hjdeheer](https://www.drivendata.org/users/hjdeheer/), [jaspervanselm](https://www.drivendata.org/users/jaspervanselm/), [JeffLim](https://www.drivendata.org/users/JeffLim/), [tolgakopar](https://www.drivendata.org/users/tolgakopar/) | 0.7332 | 0.7237 | Custom ensemble selection using weighted averaging on raw model logits. Models trained using VGG-based UNet, SwinUNetR, and ConvNext architectures. XGBoost used to generate features. Threshold fitted after weighed ensembling.
2   | [xultaeculcis](https://www.drivendata.org/users/xultaeculcis/)| 0.7318 | 0.7210 | Ensemble of UNet and EfficentNet models. Weighted sample of custom tile types like "mostly water" per epoch. Generated 17 extra spectral indices as features.
3   | [ouranos](https://www.drivendata.org/users/ouranos/) | 0.7296 | 0.7247 | Ensemble using UNet models with mit_b1, mit_b2, mit_b3 and mit_b4 backbones, encoders from segmentation models pytorch library, as well as UperNet models with convnext_tiny and convenxt_base encoders. Final models used the SWIR, NIR and Green channels. In post-processing, a threshold turned probabilities to masks and a land-sea mask from last image channel was applied to remove predicted masks on land.
Top MATLAB User | Team KELPAZKA: [kaveh9877](https://www.drivendata.org/users/kaveh9877/), [AZK90](https://www.drivendata.org/users/AZK90/) | 0.7176 |0.7044 | An ensemble of 19 U-Net convolutional neural network models trained using MATLAB. Used different U-Net structures with different number of layers, and several loss functions such as dice loss and focal loss.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Winners announcement: [Meet the Winners fo the Kelp Wanted Challenge](https://drivendata.co/blog/kelp-wanted-winners)**

**Benchmark blog post: [Kelp Wanted Challenge - Benchmark](https://drivendata.co/blog/kelp-wanted-benchmark)**