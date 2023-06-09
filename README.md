# HandwrittenLetters
## CS 6375.002 Final Project
### Project Title: Recognizing Handwritten Letters using Neural Network Algorithms
### Team Members:
- Michelle Kelman
- Jihan Wang
### Dataset Type: 
- Kaggle - https://www.kaggle.com/datasets/crawford/emnist
### Algorithm Approach:
- Own Implementation
- Neural Network models with different activation functions
- Simple PCA and t-SNE methods with 2D and 3D plot functions
### How To Use:
1. Clone our project
2. Download the dataset from the Kaggle link above
3. Copy the extracted data files into the project data folder
4. Process the data: `python data.py`
5. Run the different models:

| **Activation Function**                 | **Logistic (Sigmoid)** | **ReLU**           | **Hyperbolic Tangent** |
| :-------------------------------------: | :--------------------: | :----------------: | :--------------------: |
| **1 Hidden Layer Neural Network**       | `python logistic-1.py` | `python relu-1.py` | `python tanh-1.py`     |
| **scikit Neural Network MLPClassifier** | `python logistic-s.py` | `python relu-s.py` | `python tanh-s.py`     |
| **Keras Convolutional Neural Network**  | `python logistic-k.py` | `python relu-k.py` | `python tanh-k.py`     |

Additional hidden layer models:

| **Activation Function**                 | **Logistic (Sigmoid)**       | **ReLU**                 | **Hyperbolic Tangent**       |
| :-------------------------------------: | :--------------------------: | :----------------------: | :--------------------------: |
| **2 Hidden Layer Neural Network**       | `python extra/logistic-2.py` | `python extra/relu-2.py` | `python extra/tanh-2.py`     |
| **3 Hidden Layer Neural Network**       | `python extra/logistic-3.py` | `python extra/relu-3.py` | `python extra/tanh-3.py`     |

6. Get the data in the hidden layer and output layer: run forward() function
7. Run the PCA or t-SNE models 