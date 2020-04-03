import matplotlib
import matplotlib.pyplot as plt
import numpy as np

DI = [0.1103742, 0.05473, 0.03620, 0.02256, 0.017624, 0.01624, 0.0084, 0.0131863, 0.0068]
Acc = [0.74351, 0, 0, 0.73493,0.78176,0.78119, 0.78424,  0.78300, 0.77556]
AUC = [0.801722, 0.8084, 0.8052, 0.7992, 0.8365,  0.8422, 0.83255,  0.8367, 0.83144]
dens = [0.07236,  0.12171, 0.1469, 0.1715, 0.2209, 0.27146, 0.3543266, 0.3958, 0.42073]

"""
*********
Knn = 3 
*********

Laplace - reg = 1
DI = 0.26515
Accuracy = 0.72129
AUC = 0.785459

Laplace - reg = 5
DI = .2728252
Accuracy = 0.72215
AUC = 0.7933122

Laplace - reg = 10
DI = 
Accuracy = 
AUC = 

*********
knn = 10 
*********

Laplace - reg = 10
DI = 0.2777466
Accuracy = 0.70145
AUC = 0.7563024

Laplace - reg = 5
DI = 0.2673
Accuracy = 0.736
AUC = 0.80436

Laplace -reg = 100
DI = 0.24832753545
Accuracy = 0.73407
AUC = 0.8025167

*******
knn = 5
*******
Laplace - reg = 10
DI = 0.243475
Accuracy = 0.76974
AUC = 0.838844

Laplace - reg = 1
DI = 0.2873725
Accuracy = 0.69229
AUC = 0.75520286

"""
