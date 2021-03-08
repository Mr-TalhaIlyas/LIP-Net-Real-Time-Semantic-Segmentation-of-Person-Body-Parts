# LIP Net: Real-Time Semantic Segmentation of Person Body Parts

This repo explains the working of our Unsupervised Segemntaion Algorithm published in  **_Electronics (MDPI)_** journal.

You can access the [full paper here](https://www.mdpi.com/2079-9292/9/3/383/htm)

## Abstract

Human visual understanding and pose estimation in wild scenarios is one of the fundamental tasks of computer vision. Traditional deep convolution networks (DCN) use pooling or subsampling layers to increase the receptive field and to gather larger contextual information for better segmenting human body parts. But these subsampling layers reduce the localization accuracy of the DCN. In this work, we propose a novel DCN, which uses artuous convolution with different dilation rates to probe the incoming feature maps for gathering multi-scale context. We further combine a gating mechanism which recalibrates the convolutional feature responses adaptively by learning the channel-wise statistics. This gating mechanism helps to regulate the flow of salient features to the next stages of network. Hence our architecture can focus on different granularity from local salient regions to global semantic regions, with minimum parameter budget. Our proposed architecture achieves a processing speed of 49 frames per second on standard resolution images.

### Network Architecture

Following images shows Complete Network Architecture: Here, contrast and texture enhancement **(CTE)-Block represents** the contrast and texture enhancement block, **CE Loss** is the cross-entropy loss, complete architecture of **SE-Block** is explained in Figure 2, black arrows show the forward pass, and blue arrows show the backward error propagation.
For details on architecture kindly [visit here](https://www.mdpi.com/2079-9292/9/3/383/htm)

[]netrwork image

### Algorithm 1 (Unsupervised Semantic Segmentation)

[]

### Algorithm 2 (K nearest neighbour)

This flow chart shows how we find the value of **K** for K-means clustering algorithm, implementation is provided in code.
[]

### Example Output

Segmentation results. From top to bottom: (a) Original images, (b) segmentation results of proposed algorithm.

[]