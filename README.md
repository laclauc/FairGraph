# fairOT-embedding
Code for the AISATS 2021 article : [All of the Fairness for Edge Prediction with Optimal Transport](http://proceedings.mlr.press/v130/laclau21a/laclau21a.pdf).

-------------------------------------------------------------------------------------------------
      
CONTENT

	1. PREAMBLE
	2. ABOUT
	3. REQUIREMENTS
	4. DETAILS
	  1. Demo on synthetic graphs
	  2. Train node embeddings with Node2Vec 
	  3. Evaluate the quality of the node embeddings for the task of edge prediction

  
  1. PREAMBLE
  
This work  presents the contribution of the article 
"All of the Fairness for Edge Prediction with Optimal Transport" accepted to AISTATS 2021, where we study the problem of fairness for the task of edge prediction in graphs. 

  2. ABOUT and CITATION 
 
This repository contains source code to efficiently repair the adjacency matrix of a graph by aligning the joint distributions of nodes appearing in different sensitive groups; the script also provides a version of the original algorithm with additional individual fairness constraints. It also provides a script to train node embedding with Node2Vec (any other embedding methods could have been used) on the repaired graph and evaluate the impact for the task of edge prediction. 
  

If you use this repository, please cite:

	@inproceedings{laclau2021fairGraph,
	  title     = {All of the Fairness for Edge Prediction with Optimal Transport},
	  author    = {Laclau, Charlotte and Redko, Ievgen and Choudhary, Manvi and Largeron, Christine},
	  booktitle = {Proceedings of AISTATS},
	  year      = {2021},
	  doi       = {10.18653/v1/D17-1024},
	  pages     = {254--263},
	}
  
  3. REQUIREMENTS
  
  Required packages can be installed with the following command in your terminal :
  
  ```
  pip install -r requirements.txt
  ```
  
  4. DETAILS
  
  * `demo-synthetic.ipynb` : reproduce results from the [Appendix](http://proceedings.mlr.press/v130/laclau21a/laclau21a-supp.pdf) of the paper. 
  * `demo-polblogs.py` : IN PROGRESS 
  
  
