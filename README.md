# FairGraph

Implementation for our  paper "Fair Edge Prediction".
The code in the repository implements the emd and the regularized models of our paper. 
The implementations heavily rely on: 
- [Python Optimal Transport (POT)](https://github.com/rflamary/POT) for calculating the wasserstein distances
- [Node2Vec](https://github.com/aditya-grover/node2vec) for learning the embedding on the graph
- [networkx](https://pypi.org/project/networkx/) for handling the graph
- [Scikit-learn](http://scikit-learn.org/stable/)  for testing the prediction ability

## Relevant papers
If using the code, please cite our [paper](): 
```
@article{laclaufair2020,
  title={Fair Edge Prediction},
  author={Laclau, Charlotte and Redko, Ievgen and Choudhary, Manvi and Largeron, Christine},
  journal={xxx},
  year={2020}
}
```

