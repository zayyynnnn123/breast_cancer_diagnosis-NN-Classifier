import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt 


filename = 'wdbc.data'

losslist = []

df = pd.read_csv(filename , header = None)
#to map M = 1  and B = 0
df[1] = df[1].map({'M': 1, 'B': 0})

y_true = df.iloc[:, 1].to_numpy()

input_array = df.iloc[:, 2:].to_numpy()

#now we have the input array of size (569 , 30) which is exactly correct and expected from the data
#we also go the labels in y_true (569 ,). the index for y_true will match the corresponding sample

#we start by making the weights matrices
# 3 Layer NN
# 1st layer has 20 neurons lets say 

rng = np.random.default_rng()

# Create a 20x30 array with integers from 1 to 100 (101 is exclusive)

weights_1 = rng.uniform(low=-0.1 , high=0.1, size=(20, 30))

#(20 , 20)
weights_2 = rng.uniform(low=-0.1 , high=0.1, size=(20, 20))

#(20 , 1)
output_w =  rng.uniform(low=-0.1 , high=0.1, size=(1, 20))

#learning rate
n = 0.1

#biases 
bias_1 = np.zeros(20)

bias_2 = np.zeros(20)

bias_out = np.zeros(1)

#activation function

def RelU(x):
    return np.maximum(0 , x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def cross_entrophy_loss(y_hat , y_true):
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -((y_true * math.log(y_hat)) +((1 - y_true) * math.log(1 - y_hat)))

def D_RelU(x):
    return (x > 0).astype(float)

X_train = input_array[:400]

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / std
X_test = (input_array[400:] - mean) / std

#forward pass z = wx + b
#for training going till 399 as it is 70% of the total samples. rest 30% for testing..
for i in range(400):
    input = X_train[i]
    hidden1_z = (weights_1 @ input) + bias_1
    hidden1_a = RelU(hidden1_z)

    #Hidden Layer 2

    hidden2_z = (weights_2 @ hidden1_a) + bias_2
    hidden2_a = RelU(hidden2_z)

    #Output Layer

    y_z = (output_w @ hidden2_a) + bias_out
    y_hat = sigmoid(y_z).item()

    #forward pass complete
    #lets see the output
    if(y_hat >= 0.5):
        pred = 'M'
    else: pred = 'B' 

    if(y_true[i] == 1):
        val = 'M'
    else : val = 'B'
    loss = cross_entrophy_loss(y_hat , y_true[i])

    print("Y TRUE : ", val)
    print("Y PREDICT : ", pred)
    print("LOSS: ", loss)
    losslist.append(loss)

    #now for the training.

# OUTPUT LAYER

    delta3 = y_hat - y_true[i]

    DL_DW3 = np.outer(np.array([delta3]), hidden2_a)
    DL_DB3 = delta3


# HIDDEN LAYER 2

    delta2 = (output_w.T.flatten() * delta3) * D_RelU(hidden2_z)

    DL_DW2 = np.outer(delta2, hidden1_a)
    DL_DB2 = delta2


# HIDDEN LAYER 1

    delta1 = (weights_2.T @ delta2) * D_RelU(hidden1_z)

    DL_DW1 = np.outer(delta1, input)
    DL_DB1 = delta1
    
    
#all gradients calculated , now we update the weights and biases
#W_new = W_old - nDL/DW

# OUTPUT LAYER
    output_w = output_w - ((n)*(DL_DW3))
    bias_out = bias_out - ((n)*(DL_DB3))

#HIDDEN LAYER 2
    weights_2= weights_2 - ((n)*(DL_DW2))
    bias_2 = bias_2 - ((n)*(DL_DB2))

#HIDDEN LAYER 3
    weights_1= weights_1 - ((n)*(DL_DW1))
    bias_1 = bias_1 - ((n)*(DL_DB1))    
    
#all the parameters have been updated. 
#TRAINING IS NOW COMPLETE 
print("TRAINING COMPLETE USING THE 70% OF THE SAMPLES")

count = 0
TP = 0
FP = 0
FN = 0
TN = 0

#NOW FOR TESTING
print("BEGINNING TESTING...")
print("\n")
for x in range(400 , 569):
    input = X_test[x - 400]
    hidden1_z = (weights_1 @ input) + bias_1
    hidden1_a = RelU(hidden1_z)

    #Hidden Layer 2

    hidden2_z = (weights_2 @ hidden1_a) + bias_2
    hidden2_a = RelU(hidden2_z)

    #Output Layer

    y_z = (output_w @ hidden2_a) + bias_out
    y_hat = sigmoid(y_z).item()

    #forward pass complete
    #lets see the output
    if(y_hat >= 0.5):
        pred = 'M'
        pred_val = 1
    else: 
        pred = 'B'
        pred_val = 0  

    if(pred_val == y_true[x]):
        count = count + 1

#TP 
    if pred_val == 1 and y_true[x] == 1:
        TP += 1
    elif pred_val == 1 and y_true[x] == 0:
        FP += 1
    elif pred_val == 0 and y_true[x] == 1:
        FN += 1
    else:
        TN += 1

    if(y_true[x] == 1):
        val1 = 'M'
    else : val1 = 'B'
    
    print("Y TRUE : ", val1)
    print("Y PREDICT : ", pred)
    print("LOSS: ", cross_entrophy_loss(y_hat , y_true[x]))

# Accuracy
# total samples are 400 - 569 = 169 , ther
# efore accuracy = count/169 * 100 
loss_array = np.array(losslist)

xpoints = np.arange(len(loss_array))


plt.plot(xpoints, loss_array)
plt.title("Loss vs iterations")
plt.savefig("loss.png")


acc = (count/169) * 100

print ("the accuracy of the Network is : ", acc)

#F-measure
#we need TP , FP , FN , TN
if TP + FP != 0:
    precision = TP / (TP + FP)
else:
    precision = 0

if TP + FN != 0:
    recall = TP / (TP + FN)
else:
    recall = 0

if precision + recall != 0:
    F1 = 2 * ((precision * recall) / (precision + recall))
else:
    F1 = 0

print("the F1 measure is : ", F1)
print("precision = ", precision )
print("recall = ", recall )
plt.show()



