

<img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/><img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" />
# LIP Net: Real-Time Semantic Segmentation of Person Body Parts

This repo explains the working of our Real-Time Semantic Segmentation of Person Body Parts published in  **_ICROSS 2020_** conference.

You can access the [full paper here](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE09410358&mark=0&useDate=&bookmarkCnt=0&ipRange=N&accessgl=Y&language=ko_KR)

## Abstract

Human visual understanding and pose estimation in wild scenarios is one of the fundamental tasks of computer vision. Traditional deep convolution networks (DCN) use pooling or subsampling layers to increase the receptive field and to gather larger contextual information for better segmenting human body parts. But these subsampling layers reduce the localization accuracy of the DCN. In this work, we propose a novel DCN, which uses artuous convolution with different dilation rates to probe the incoming feature maps for gathering multi-scale context. We further combine a gating mechanism which recalibrates the convolutional feature responses adaptively by learning the channel-wise statistics. This gating mechanism helps to regulate the flow of salient features to the next stages of network. Hence our architecture can focus on different granularity from local salient regions to global semantic regions, with minimum parameter budget. Our proposed architecture achieves a processing speed of 49 frames per second on standard resolution images.

### Network Architecture

Following images shows Complete Network Architecture: Complete Network architecture of LIP-Net. Here SER represents SE-ResNet Block, ‘s’ and ‘d’ are stride and dilation rate respectively. Si represents the different stages of network, where i∈{1,2,3,6}. [visit here](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE09410358&mark=0&useDate=&bookmarkCnt=0&ipRange=N&accessgl=Y&language=ko_KR)

![alt text](https://github.com/Mr-TalhaIlyas/LIP-Net-Real-Time-Semantic-Segmentation-of-Person-Body-Parts/blob/master/screens/img1.png)
### LIP Dataset
For details of LIP dataset [visit here](https://github.com/Mr-TalhaIlyas/Color-Pallets-and-Class-Names-for-Semantic-Segmentation-Datasets)

### Quantative Results
![alt text](https://github.com/Mr-TalhaIlyas/LIP-Net-Real-Time-Semantic-Segmentation-of-Person-Body-Parts/blob/master/screens/img2.png)

### Example Output

![alt text](https://github.com/Mr-TalhaIlyas/LIP-Net-Real-Time-Semantic-Segmentation-of-Person-Body-Parts/blob/master/screens/img3.png)
