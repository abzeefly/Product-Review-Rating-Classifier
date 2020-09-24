#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

##############################################################################################
#Name : Abdullah Zaid Ansari    
#StudentID : z5099229
#Assignment : hw2
##############################################################################################

#Briefly describe how your program works, and explain any design and training decisions you made along the way
""" 
    So for my program I made the following changes:
    
    1.Preprocessing : I did not add any pre-processing as the training and test data 
    could be different and I wanted to make a more neural network modle than a 
    machine learning one.

    2. PostProcessing : I just saved the vocab size for embedding in this phase.

    3. ConvertLabel: I converted labels from [1, 5] to [0,4] so I could put them through my Loss function.

    4. ConvertNetOutput : In this step I take the argmax of the probabilities for a single
     single sentence to be classified into ([0,4] and then added 1 to fil the labels into the desired target.

     I took a classiication approach rather than a regression approach for this project as without having sentiment
     I found it difficult to guage the probabilities into the right classes. 

    5. Net : For my neural netwrok I had an Embedding -> LSTM --ReLu-->Linear(hidden)--ReLu--> Linear(target)--Dropout netwrok.
        
        Embedding Layer -> To create a dense vector of words and its closest relations from the training set. I did this because it 
        make the model more relevant to the training data, using glove would have been better as it would give me better word relation
        matrix but using it was bit of a challenge so I just used my embedding from the traiing vocabulary.

        LSTM : I used LSTM to make my model into a RNN and learn long term dependencies between words.

        Linear Layers : Just to have a step to insert activation functions to use my loss functions and also 
        create a buffer between numerous layers from lstm to target layers.

        Dropout : I zero out any p = 0.2 to reduce confusion between target classes.
    
    6.Loss: I used a simple NLLLoss() function as it is can be used for mulitclass classification.

    trainValSplit = 0.8
    batchSize = 32
    epochs = 5
    optimiser = toptim.SGD(net.parameters(), lr=0.01)

    I used these values because at this learning rate the model can work well with loss function and the gradient reduces and
    doesnt explode.
    
    I am sure if I used more epoches the program would even better but the wait time.

"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
from torch.nn import BCELoss

import numpy as np
# import sklearn

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################
vocab_size = 0
embedding_matrix = 0
def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    vocab_size =len(vocab)

    # embedding_dim = 50
    # embedding_matrix = np.zeros((vocab_size,embedding_dim))
    # i = 0
    # for word in batch:
    #     embedding_vector = embedding_index.get(word)
    #     if embedding_vector is not None:
    #         embedding_matrix[i] = embedding_vector
    #     i+=1

    return batch

stopWords = {}


wordVectors = GloVe(name='6B', dim=50)

 
# for line in wordVectors:
#     val = line.split(1)
#     word = val[0]
#     coefs = np.asarray(val[1:], dtype='float32')
#     embedding_index[word] = coefs 
# f.close()



###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    # datasetLabel.unsqueeze(1)
    # print(datasetLabel.size())
    datasetLabel = (datasetLabel -1).long()
    return datasetLabel

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """

    # print(netOutput)
    softmax = tnn.Softmax(dim = 1)
    netOutput = softmax(netOutput)
    netOutput = (np.argmax(netOutput, axis = -1)) + 1
    # print(netOutput)
    return netOutput.float()


###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """
    def __init__(self):
        super(network, self).__init__()
        self.input_size = 10
        self.hidden_size = 32
        self.output_size = 5
        self.embedding_dim = 32

        self.embedding = tnn.Embedding(vocab_size, self.embedding_dim, padding_idx =0 )
        # self.lstm = tnn.LSTM(self.embedding_dim, self.hidden_size)
        # self.lin1 = tnn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lin2 = tnn.Linear(self.hidden_size, self.output_size)

        self.relu = tnn.ReLU()
        self.dropout = tnn.Dropout(p=0.1)
        self.softmax = tnn.Softmax(dim = 1)


    def forward(self, input, length):
        
        #creating an embed
        input_size = input.shape[1] * input.shape[2]
        # print(input.size())
        output = input.view(input.size(0),-1).long()
        embed = self.embedding(output)
        # output = input.reshape(1,input.size(0),-1)
        # print(output.size())
        
        lstm = tnn.LSTM(self.embedding_dim, self.hidden_size)
        #passing the embed to the lstm
        lstm_layer, (hidden,cell) = lstm(embed)
        lstm_layer = self.relu(lstm_layer)

        #passing the lstm to linear layers
        lstm_layer = lstm_layer.reshape(input.size(0),-1)
        lin1 = tnn.Linear(lstm_layer.size(1), self.hidden_size)
        linear_layer1 = lin1(lstm_layer)
        linear_layer1 = self.relu(linear_layer1)

        #creating the target layer with dropout
        linear_layer2 = self.lin2(linear_layer1)
        target_layer = self.dropout(linear_layer2)
        return target_layer

class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.main = tnn.LogSoftmax(dim = 1)

    def forward(self, output, target):
        nllloss = tnn.NLLLoss()
        # print(output.size())
        # print(target.size())
        nnLoss = nllloss(output,target.long())
        return nnLoss



net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""

lossFunc = tnn.NLLLoss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 3
optimiser = toptim.SGD(net.parameters(), lr=0.001)
