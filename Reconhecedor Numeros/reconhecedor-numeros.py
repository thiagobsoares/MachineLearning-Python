from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer
import numpy as np
import csv
import random

ds = SupervisedDataSet(400, 10)
matriz = [];

cr = csv.reader(open("numeros-pixel.csv","rb"))
print "Lendo os arquivos"
for row in cr:
    target = int(float(row[-1]))
    pixel = row[0:len(row) - 1]
    arrayTarget = [0.0] * 10
        
    if target == 10:
        arrayTarget[0] = 1
    else:
        arrayTarget[target] = 1

    aux = np.append(pixel,arrayTarget)
    matriz.append(aux)

print "Finalizou leitura.\nMatriz formada: %d por %d\n\nIniciado o embaralhamento da matriz" % (len(matriz), len(matriz[0]))
random.shuffle(matriz)

porcentagem_treino = 0.9
total_dado = len(matriz)
total_treino = int(total_dado * porcentagem_treino)
matriz_treino = matriz[0:total_treino]
matriz_validacao = matriz[total_treino:total_dado]

print "Matriz separada\n%d matrizes para treino\n%d matrizes para validacao" % (len(matriz_treino), len(matriz_validacao))
for line in matriz_treino:
    ds.addSample(line[0:400], line[400:410])
    
rede = buildNetwork(400,25,10, bias=True, outclass=SigmoidLayer, hiddenclass=SigmoidLayer)
trainer = BackpropTrainer(rede,ds)

print "Iniciando treinamento"
for i in range(50):
    print "Treino %d: %f" % (i, trainer.train())

del line
print "Iniciando validacao"
acerto = 0
for line in matriz_validacao:
    retorno = rede.activate(line[0:400]).argmax(axis=0)
    valor = line[400:410].argmax(axis=0)
    retorno_transformado = 10 if retorno == 0 else retorno
    valor_transformado = 10 if valor == 0 else valor
    if(retorno_transformado==valor_transformado): acerto += 1
    print "Rede: %d   --- Valor Correto: %d" % (retorno_transformado, valor_transformado)
print "Acertos: %d" % (acerto)
