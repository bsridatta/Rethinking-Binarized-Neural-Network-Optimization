---   
<div align="center">    
 
# Rethinking Binarized Neural Network Optimization     


[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)

[![Paper](http://img.shields.io/badge/paper-arxiv.1906.02107-B31B1B.svg)](https://arxiv.org/pdf/1906.02107.pdf)

</div>
 
## Description   

This repository aims to reproduce the results in "Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization" as part of the NeurIPS 2019 reproducibility challenge. 
We have implemented the Binary optimization algorithm in PyTorch and with it are able to train a binary neural network on CIFAR-10. 
See the [reproducibility report](report/reproducibility_report.pdf) for details. 

## How to run

First, install dependencies   

```bash
# clone project   
git clone https://github.com/nikvaessen/Rethinking-Binarized-Neural-Network-Optimization

# install project   
cd https://github.com/nikvaessen/Rethinking-Binarized-Neural-Network-Optimization
pip install -e .   
pip install requirements.txt
 ```   

If you are interested in training a BNN on cifar-10, you can navigate to `research_seed/cifar` and run `cifar_trainer.py`.   

 ```bash
# module folder
cd research_seed/cifar/   

# run module 
python cifar_trainer.py    
```

## Main Contribution      

In order to reproduce the original paper we have implemented the following:
 
- [Bytorch](research_seed/bytorch) implements binary optimisation and binary layers in PyTorch
- [cifar](research_seed/cifar) implement BinaryNet ([from this paper](https://arxiv.org/abs/1602.02830)) for CIFAR-10
- [theoretical](research_seed/theoretical) implements experiments to disprove the approximation viewpoint as well as behaviour of learning rates under latent-weight optimisation
- [experiments](experiments) contains convenience scripts to reproduce the experiments of section 5.1 and 5.2 of the original paper 
