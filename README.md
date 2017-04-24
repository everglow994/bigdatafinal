# bigdatafinal
This is the code for Big Data 2016 Final project


------------
Introduction
------------ 
    
        In this project, a Collaborative Denoising Auto Encoder Neural Network is
    designed and implemented to give user preference prediction of Ml-100k/ Ml-1M/ Ml-10M/ Yelp Dataset


-------------
File Document
-------------

        We use keras library to establish a Collaborative Denoising Auto Encoder Neural Network with
    one hidden layer, the data is     
     
-----
Usage
-----   
First, install libraries:

```sh
pip install -r requirements.txt
```

Then,

```sh
# CPU
python train.py

# GPU
THEANO_FLAGS=device=gpu,floatX=float32 python train.py
```
----------------------
Implementation Details
----------------------

- [x] Establish CDAE model
- [x] Implement negative sampling
- [x] Collaborative Filtering 
- [x] Data Corruption
- [x] Coding to use Mean Average Precision as evaluation metric
- [x] Get results
