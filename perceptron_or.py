import numpy as np
import pandas as pd
from nn.perceptron import Perceptron


# Construction de la matrix X et du vecteur y

X = np.array([[0,0],[0,1],[1,0],[1,1]])

y = np.array([[0],[1],[1],[1]])

# definition du perceptron et entrainement
print("[INFO]  entrainement du perceptron .........")

p = Perceptron(X.shape[1], alpha=0.1)

p.fit(X,y,epochs=20)

# test du perceptron
print("[INFO] test du perceptron ....")

for (x,target) in zip(X,y):

    pred = p.predict(x)

    print("[INFO] data ={}, valeur reelle ={} et valeur predite={}".format(x,target,pred))
