# Quantum Natural Language Processing Tutorial

This tutorial provides an overview of Quantum Natural Language Processing (QNLP), a cutting-edge application of quantum computing dedicated to the execution of Natural Language Processing (NLP) tasks with the aid of quantum computers. 

We will provide a theoretical overview of NLP and QNLP. Then, we will construct two workflow examples used to train QNLP models. First, using single-threaded quantum-classical simulation. Second, as distributed computing workflows that could incorporate High-Performance Computing (HPC) resources. Finally, we will present metrics that can be used to benchmark and compare QNLP performance against that of classical NLP models. 

Attendees will participate in hands-on activities demonstrating and testing these workflows using numerical simulators and examples of real-world applications. This tutorial will help attendees train Quantum Long Short-Term Memory (QLSTM) and Compositional Distributional Categorical (DisCoCat) models for respectively tagging the words in a small dataset with their parts of speech (POS) and classifying the meaning of the sentences they form. They will also explore the Quantum Support Vector Machine (QSVM) and Quantum Discriminator (QD) models for classification tasks. 

Moreover, the tutorial will delve into quantum binary and multi-class classification of more complex datasets by benchmarking the performance of quantum classifiers with different quantum embeddings and loss functions against Convolutional Neural Networks (CNN) trained on the Frontier supercomputer at ORNL, and the other hybrid models.


## Target Audience ##

The tutorial targets a broad audience. It is geared towards individuals interested in exploring the capabilities of QNLP on quantum plus HPC systems. Attendees with no previous knowledge of QNLP will benefit the most from this tutorial, as it will ground the concepts and workflow required to simulate the execution of these tasks, and showcase simple applications by training some QNLP models. While seasoned professionals on the field may find some of the content familiar, they can expect to gain new insights about quantum classification of complex datasets in the intermediate portion of the tutorial.



## Agenda ##

### Session #1: 10:00 AM - 11:30 AM Pacific Time ###
|#  | Agenda                                    | Speaker                       | Duration | 
|---|-------------------------------------------|-------------------------------|----------|
|1. | Introduction and Welcome!                 | Team                          | 5 min    |
|2. | QNLP Workflow on HPC + QC                 | In-Saeng Suh                  | 25 min   |
|3. | Category Theory in QNLP                   | Francisco Rios                | 30 min   |
|4. | Classical NLP Modeling                    | Mayanka Chandra Shekar & John Gounley       | 25 min   |
|5. | Coming Up in Session #2...                | Eduardo Coello Perez          | 5 min    |


### Session #2: 1:00 PM - 2:30 PM Pacific Time ###
|#  | Agenda                                    | Speaker                       | Duration | 
|---|-------------------------------------------|-------------------------------|----------|
|1. | Quantum Neural Network for Text Classification | Kathleen Hamilton        | 30 min   |
|2. | DisCoCat and QLSTM models for simple tasks| Eduardo Coello Perez          | 30 min   |
|3. | Adiabatic Quantum Support Vector Machines | Prasanna Date & Dong Jun Woun | 30 min   |


## Prerequisits ##
Follow the instructions at https://conda.io/projects/conda/en/latest/user-guide/install/index.html to install conda.

Make sure your conda version is > 23.7. You can check your conda version by typing the command:

$ conda --version


## Getting Started ##
1. Open a terminal window
2. Clone the QNLP Tutorial repo: `git clone https://github.com/Tognognongo/QCE-tutorial.git`
3. Navigate to the QCE-tutorial directory: `cd QCE-tutorial`
4. Type the following commands:
   
   $ conda env create -f environment.yml
   
   $ conda activate qnlp-tut
   
   $ python3 -m ipykernel install --user --name=qnlp-tut
   
   $ jupyter lab


## Using Codespaces $$



