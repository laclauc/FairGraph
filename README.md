# fairOT-embedding
Code for the AiStat 2021 article : All of the Fairness for Edge Prediction with Optimal Transport

-------------------------------------------------------------------------------------------------
      
CONTENT

	1. PREAMBLE
	2. ABOUT
	3. REQUIREMENTS
	4. USAGE
	  1. Repair the adjacency matrix of a graph using optimal transport
	  2. Train node embeddings with Node2Vec 
	  3. Evaluate the quality of the node embeddings for the task of edge prediction
	5. AUTHOR
	6. COPYRIGHT
  
  1. PREAMBLE
  
This work  presents the contribution of the article 
"All of the Fairness for Edge Prediction with Optimal Transport" accepted to AiStat 2021, where we study the problem of fairness for the task of edge prediction in graphs. 

  2. ABOUT 
 
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
