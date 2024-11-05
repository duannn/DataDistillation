# DataDistillation
Data Distillation: Data-efficient learning framework


### mnist_synthesis.py
This file involves training ConvNet-3 on Real MNIST dataset, generation of synthetic MNIST data using Attention Matching, training ConvNet-3 on synthetic MNIST data, and test both the model trained on real dataset and the model on synthetic data on real MNIST test set. 

Task 1: 2 a), b), c), and e) for MNIST dataset are completed in this file

### mhist_synthesis.py
This file involves training ConvNet-7 on Real MHIST dataset, generation of synthetic MHIST data using Attention Matching, training ConvNet-7 on synthetic MHIST data, and test both models on real MHIST test set. <br/>
Task 1: 2 c), b), c), and e) for MHIST dataset are completed in this file

####Note: since cross_architecture.py, NAS.py, and task2.py require to load the synthetic images, mnist_synthesis.py and mhist_synthesis.py need to be run first in order to generate and store the synthetic data. 

### gaussian_mnist.py
This file carried out synthetic MNIST data generation using Attention Matching Alogrithm with random Gaussian noise initialization. <br/>
Task 1: 2 d) for MNIST dataset is completed in this file

### gaussian_mhist.py
This file carried out generation of synthetic MHIST data using Attention Matching Algorithm with random Gaussian noise initialization. <br/>
Task 1: 2 d) for MHIST dataset is completed in this file

### cross_architecture.py
This file trains AlexNet on synthetic MNIST and MHIST data generated from 2 b) and evaluate the performamce on real MNIST and MHIST test dataset, respectively. <br/>
Task 1: 3 is completed in this file

### NAS.py
This file implements Neural Architecture Search with the help of synthetic datasets generated from previous task. As in previous generation, the synthesized images are stored in folder. In this file, it will directly load the data from corresponding folders for MNIST and MHIST. The sample space contains various configurations of CNN networks. <br/>
Task 1: 4 is completed in this file

### task2.py
This file implements synthetic MNIST data generation using PAD. ConvNet-3 is used to be trained on the generated synthetic data and then evaluated on real MNIST test set. <br/>
Task 2 is completed in this file
