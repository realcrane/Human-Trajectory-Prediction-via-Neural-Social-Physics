# [Human Trajectory Prediction via Neural Social Physics](https://arxiv.org/pdf/2207.10435.pdf)
![](https://github.com/realcrane/Human-Trajectory-Prediction-via-Neural-Social-Physics/blob/main/images/model.png)

Trajectory prediction has been widely pursued in many fields, and many model-based and model-free methods have been explored. The former include rule-based, geometric or optimization-based models, and the latter are mainly comprised of deep learning approaches. In this paper, we propose a new method combining both methodologies based on a new Neural Differential Equation model. Our new model (Neural Social Physics or NSP) is a deep neural network within which we use an explicit physics model with learnable parameters. The explicit physics model serves as a strong inductive bias in modeling pedestrian behaviors, while the rest of the network provides a strong data-fitting capability in terms of system parameter estimation and dynamics stochasticity modeling. We compare NSP with 15 recent deep learning methods on 6 datasets and improve the state-of-the-art performance by 5.56%-70%. Besides, we show that NSP has better generalizability in predicting plausible trajectories in drastically different scenarios where the density is 2-5 times as high as the testing data. Finally, we show that the physics model in NSP can provide plausible explanations for pedestrian behaviors, as opposed to black-box deep learning.

## Get Started
### Dependencies
Below is the key environment under which the code was developed, not necessarily the minimal requirements:  
  
 1 Python 3.8.8  
 2 pytorch 1.9.1  
 3 Cuda 11.1  
  
And other libraries such as numpy.  
### Prepare Data  
Raw data: SDD (https://cvgl.stanford.edu/projects/uav_data/) and ETH/UCY (https://data.vision.ee.ethz.ch/cvl/aess/dataset/)  
Algorithms in data/SDD_ini can be used to process raw data into training data and testing data. The training/testing split is same as Y-net.  

### Training  
We employ a progressive training scheme. Run train_goals.py, train_nsp_wo.py and train_nsp_w.nsp to train Goals-Network, Collision-Network with k_env and CVAE respectively. The outputs are saved in saved_models. There are trained models in saved_models for test.  
For example  
`python train_goals.py                                                                                `  

### Authors  
Jiangbei Yue, Dinesh Manocha and He Wang  
Jiangbei Yue scjy@leeds.ac.uk  
He Wang, h.e.wang@leeds.ac.uk, [Personal Site](http://drhewang.com/)  
Project Webpage: http://drhewang.com/pages/NSP.html  

### Contact  
If you have any questions, please feel free to contact me: Jiangbei Yue (scjy@leeds.ac.uk)  

### Acknowledgement  
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 899739 [CrowdDNA](https://crowddna.eu/).  

### Citation (Bibtex)  
Please cite our paper if you find it useful:
```
@InProceedings{Jiang_trajectory_2022,
author={J. {Yue} and D. {Manocha} and H. {Wang}},
booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
title={Human Trajectory Prediction via Neural Social Physics},
year={2022}}
```  

### License  
Copyright (c) 2022, The University of Leeds, UK. All rights reserved.  
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:    
 1 distributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.    
 2 distributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




