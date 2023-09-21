# Importation des packages
import numpy as np
import pandas as pd

# classe Perceptron

class Perceptron:

    def __init__(self, N, alpha=0.1):
        """Initialisation de la classe

        Args:
            N (_type_): Le nombre de colonnes
            alpha (float, optional): Le learning rate par Defaults to 0.1.
        """
        self.W =  np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self,x):
        # La direction
        return 1 if x > 0 else 0

    def fit(self, X,y, epochs=10):
        """Cette fonction permet d'entrainer le modèle

        Args:
            X (_type_): Les inputs
            y (_type_): Les outputs
            epochs (int, optional): Le nombre d'epoches par Defaults to 10.
        """
        # X = np.atleast_2d(X)

        X = np.c_[X,np.ones((X.shape[0]))]

        # Parcourt des epochs

        for epoch in np.arange(0, epochs):
            if epochs >0 and epoch> 0 and (epoch+1)%epochs==0 :
                print("[INFO] traitement de {} /{}".format(epoch + 1,epochs))

            for (x, target) in zip(X,y):

                # Calculer la prediction
                p = self.step(np.dot(x,self.W))

                if p != target: # On compare la valeur predite à la valeur réelle
                    error = p - target

                    # Mise à jour des poids W

                    self.W += -self.alpha * error * x

    def predict(self,X, addBias=True):

        X = np.atleast_2d(X)

        if addBias:
            
            # Matrix
            X = np.c_[X, np.ones((X.shape[0]))]


        return self.step(np.dot(X,self.W))
        