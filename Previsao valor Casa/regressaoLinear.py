#-*- encoding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
def calcularFuncao(x, theta):
    return x.dot(theta);

def funcaoDeCusto(x, y, theta):
    m = x.shape[0]	
    soma = sum((calcularFuncao(x,theta) - y) ** 2)
    return soma / (2.0 * m); 

def normalEqn(x, y):
    novoTheta = np.zeros(x.shape[1], dtype=np.float)
    novoTheta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y);
    return novoTheta; 


def funcaoGradiente(taxa_aprendizado, x, y, theta):
    m = x.shape[0]
    novoTheta = np.zeros(x.shape[1], dtype=np.float)
    for i in range(novoTheta.shape[0]):
        soma = sum((calcularFuncao(x,theta) - y) * x[:,i])
        novoTheta[i] = theta[i] - taxa_aprendizado / m * soma
    return novoTheta

def somar(valores):
	soma = 0.0
	for v in valores:
	    soma += v
	return soma


def media(valores):
	soma = somar(valores)
	qtd_elementos = len(valores)
	media = soma / float(qtd_elementos)
	return media


def variancia(valores):
	_media = media(valores)
	soma = 0.0
	_variancia = 0.0

	for valor in valores:
	    soma += math.pow( (valor - _media), 2)
	_variancia = soma / float( len(valores) )
	return _variancia


def desvio_padrao(valores):
	return math.sqrt( variancia(valores) )
 

df = pd.read_csv("dados-valor-casa.csv");

x_df = df[['x1','x2']];
y_df = df['y'];


x = x_df.values
y = y_df.values

linearizar = False

if (linearizar):
	for j in range(x.shape[1]):
		desvioP = desvio_padrao(x[:,j])
		mediaU = media(x[:,j])
		for i in range(len(x[:,j])):
			x[i,j] = (x[i,j] - mediaU) / desvioP
		del desvioP
		del mediaU

ones = np.ones(len(x), dtype=np.int)
x = np.array([ones, x[:,0], x[:,1]]).T



print normalEqn(x,y)
#print("\n\n\nf(x) = theta0 + theta1 * x")
#print("Parametros = theta0, theta1")
#print("Funçao de custo = J(theta0, theta1) = 1/(2*m) * sum((f(x¹) - y¹)²)")
#print("Funçao de gradiendo =  theta0 - taxa_aprendizado * (1/m) * sum((f(x¹) - y¹) * x0)")
#print("Objetivo e minimizar funçao de custo")

taxa_aprendizado = 0.005;

#Indice 0 = theta0, Indice 1 = theta1...

theta = np.zeros(x.shape[1], dtype=np.float)

for i in range(0):
    print theta
    print "Funcao de Custo: %f" % (funcaoDeCusto(x,y,theta))
    theta = funcaoGradiente(taxa_aprendizado,x,y,theta);

#print (calcularFuncao(3.5,theta))
#print (calcularFuncao(7.0,theta))


#plt.scatter(x,y);
#plt.show();

