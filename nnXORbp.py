import random
import time
import math

class Neuron:
    weigth = False
         
    def set_inputs(self, inputs):
        self.inputs = inputs
        if not self.weigth:
            self.weigth = [random.uniform(-0.2,0.2) for i in inputs]
    
    def sigmoid(self, x):
        return math.tanh(x)
         
    def output(self):
        out = 0.0
        for i in range(len(self.inputs)):
            out += self.inputs[i]*self.weigth[i]
        out = self.sigmoid(out)
        return out

          
class NeuralNet:

    rate = 0.5
    momentum = 0.1

    def __init__(self, num_input, num_hidden, num_output):
        self.i_out = [0.0]*num_input
        self.h_out = [0.0]*num_hidden
        self.o_out = [0.0]*num_output
        self.desired_out = [0.0]*num_output
        
        self.hidden = []
        for h in range(num_hidden):
            n = Neuron()
            self.hidden.append(n)
        self.output = []
        for o in range(num_output):
            n = Neuron()
            self.output.append(n)
            
    def set_input(self, input):
        self.i_out = input
        
    def set_desired_output(self, output):
        self.desired_out = output
        
    def update(self):
        for h in range(len(self.hidden)):
            self.hidden[h].set_inputs(self.i_out)
            self.h_out[h] = self.hidden[h].output()
            
        for o in range(len(self.output)):
            self.output[o].set_inputs(self.h_out)
            self.o_out[o] = self.output[o].output()
        
        return self.o_out
        
    def dsigmoid(self, y):
        return 1.0 - y**2

    def back_propagate(self):
        #calculating output deltas
        output_deltas = [0.0]*len(self.output)
        for o in range(len(self.output)):
            error = self.desired_out[o]-self.o_out[o]
            output_deltas[o] = self.dsigmoid(self.o_out[o])*error
        #calculating hidden deltas
        hidden_deltas = [0.0]*len(self.hidden)
        for h in range(len(self.hidden)):
            error = 0.0
            for o in range(len(self.output)):
                error = error + output_deltas[o]*self.output[o].weigth[h]
            hidden_deltas[h] = self.dsigmoid(self.h_out[h])*error
    
        for o in range(len(self.output)):
            for h in range(len(self.hidden)):
                change = output_deltas[o]*self.h_out[h]
                self.output[o].weigth[h] = self.output[o].weigth[h]+self.rate*change
                
        for h in range(len(self.hidden)):
            for i in range(len(self.i_out)):
                change = hidden_deltas[h]*self.i_out[i]
                self.hidden[h].weigth[i] = self.hidden[h].weigth[i]+self.rate*change
                
        error = 0.0
        for o in range(len(self.o_out)):
            error += 0.5*(self.desired_out[o]-self.o_out[o])**2
        return error
    
    def train(self, patterns, iterations=10000):
        for i in range(iterations):
            error=0.0
            for p in patterns:
                self.set_input(p[0])
                self.set_desired_output(p[1])
                self.update()
                error = error + self.back_propagate()
                if i % 100 == 0:
                    print('error %-.5f' % error)
        
    def test(self, patterns):
        for p in patterns:
            self.set_input(p[0])
            print(p[0], '->', self.update())
        
def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]],
    ]
    n = NeuralNet(2,2,1)
    n.train(pat)
    n.test(pat)


if __name__ == '__main__':
    demo()
