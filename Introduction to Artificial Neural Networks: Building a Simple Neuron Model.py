#importing required libraries
import numpy as np
import matplotlib.pyplot as plt

#define the sigma activation function
def sigmoid(x):
  return 1/(1+np.exp(-x))

#define a simple artificial neuron
class ArtificiallNeuron:
  def __init__(self, num_inputs):
    #Random initialization of weights and bias
    self.weights = np.random.rand(num_inputs)
    self.bias = np.random.rand()

  def forward(self,inputs):
      #weighted sum of inputs and bias
      weighted_sum = np.dot(self.weights,inputs) + self.bias
      #apply the activation function
      output = sigmoid(weighted_sum)
      return output

#example usage
#Number of inputs for the neuron

num_inputs = 2
neuron = ArtificiallNeuron(num_inputs)
#Create the instance of the neuron
inputs = np.array([0.5, 0.8])

#Calculate output of the neuron
output = neuron.forward(inputs)

print("Neuron's output:", output)
