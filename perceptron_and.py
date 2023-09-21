import numpy as np

from nn.perceptron import Perceptron

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])

p = Perceptron(X.shape[1], alpha=0.1)

print("[INFO] entrainement du perceptron ...")

p.fit(X, y, epochs=25)

print("[INFO] Teste du perceptron ....")

for (x, target) in zip(X,y):

    pred = p.predict(x)

    print("[INFO] data={}, valeur reelle={} et valeur predite={}".format(x,target,pred))
