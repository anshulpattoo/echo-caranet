<img width="420" alt="QMIND_Research_Logo" src="https://github.com/anshulpattoo/echo-caranet/assets/41569741/cbb33345-36fa-455c-973f-07739b0311b9">

__Implementation of the CUCAI 2022 paper, Automated Video Segmentation on Ultrasound for Cardiac Disease Intervention__

## Abstract

![example_2](https://github.com/anshulpattoo/echo-caranet/assets/41569741/aff91c04-e463-4574-a809-bb62c20a5c8d)

Cardiovascular disease is the global leading cause of death, accounting for nearly 40% of all deaths in Canada. Visualization of the left ventricle allows for assessment of the ejection fraction metric, a measurement of the percentage of blood that leaves the heart in each contraction, informing physicians about the current severity of the condition and the approach that should be taken towards a treatment plan. Unfortunately, human assessment of ejection fraction is an arduous task for the clinician and is subject to error due to the inaccuracies in segmentation of the ventricle. We implement and train a U-Net-based deep learning model on anonymized patient data to develop a method for automatically generating accurate left ventricle segmentations on echocardiogram ultrasounds. We quantitatively evaluate our model performance against expert-segmented ground truths and achieve a validation mean Intersection over Union (IoU) score of 0.84 and mean Dice similarity coefficient of 0.91 after two epochs of training. A live video demonstration of system performance is available here. Through this research, we successfully implement an architecture for generating anatomically accurate left ventricle segmentations on ultrasound. 

## CaraNet Architecture

CaraNet, developed by Lou et al., was adapted to address segmentation of the left ventricle for the end-systole and end-diastole frames. CaraNet applies context and axial reverse attention operations to detect global and local feature information, as well as a parallel partial decoder mechanism to produce a high-level semantic global map [1]. 

<img width="468" alt="Picture1" src="https://github.com/anshulpattoo/echo-caranet/assets/41569741/699450d5-ca66-4fee-8083-47f8065418af">

High-level overview of CaraNet architecture [1].

The architecture fundamentally subscribes to a U-Net structure with the two aforementioned major mechanisms that aim to address the concern of segmentation of small medical objects. {fi | i = 0,...,n}, where n represents the depth of the network is a set of downsampling operations. The Channel-wise Feature Pyramid (CFP) module is a network which applies a series of dilated convolution, a specialized form of the standard 3x3 convolution which inserts gaps between pairs of convolution elements, channels to extract effective features [2]. Axial Reverse Attention (A-RA) is a technique that attempts to recognize salient or highlight features for a given activation map. A parallel partial decoder is a technique that aggregates activation maps from different stages of downsampling operations.

CaraNet is trained using deep supervision, which sums individual losses from distinct stages in CaraNet. This individual loss function L applies a weighted intersection over union (IoU) and weighted binary cross-entropy (BCE) loss function for both global loss and local (pixel-level) loss, respectively. The individual function measures the loss for three side-outputs (S3, S4, S5) and the global map Sg. The resultant total loss is measured as follows: 

<img width="246" alt="Picture2" src="https://github.com/anshulpattoo/echo-caranet/assets/41569741/c7e7cb83-d29d-4968-bed4-e7249815115d">

## Running this implementation

The dataset used for this implementation is a large set of echocardiogram videos from Stanford Artificial Intelligence and Medical Imaging (AIMI). This set can be accessed [here](https://echonet.github.io/dynamic/). 

The video directory entitled `a4c-video-dir` should be saved in the base directory of this repository. 

This implementation was developed in a Linux OS, running Ubuntu 20.04.6. A conda environment was produced for this implementation, with all dependencies indicated in `CaraNet_Env.yml`.

While the conda environment is activated, in the base directory, execute `python train.py` from the command line. This script executes the visualization automatically as well. 

Please note that our team's work consisted of reviewing relevant methods to approach echocardiogram segmentation and adapting the original implementation of CaraNet towards our problem. Our adaptations primarily involved modifying the original CaraNet implementation to ingest data from the set indicated above, during training. Therefore, comments and some codes may not be directly relevant to this specific project, `echo-caranet` (but rather, the original CaraNet).

## References (IEEE)

[1] Lou A, et al. CaraNet: context axial reverse attention network for segmentation of small medical objects, Proc. SPIE 12032, Medical Imaging 2022: Image Processing, 120320D (4 April 2022); https://doi.org/10.1117/12.2611802.

[2] Lou A and Loew M., CFPNET: Channel-Wise Feature Pyramid For Real-Time Semantic Segmentation, 2021 IEEE International Conference on Image Processing (ICIP), 2021, pp. 1894-1898, https://doi.org/10.1109/ICIP42928.2021.9506485. 
