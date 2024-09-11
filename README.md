# 
<div align="center">

<h1> k-Redundancy Graph Neural Networks </h1>

![python-3.11](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch-2.2.0](https://img.shields.io/badge/torch-1.13.1-orange)
![release-version](https://img.shields.io/badge/release-0.1-green)
![license](https://img.shields.io/badge/license-GPL%202-red)
</div>


## On the Two Sides of Redundancy in Graph Neural Networks
This repository contains the code for paper [On the Two Sides of Redundancy in Graph Neural Networks](https://link.springer.com/chapter/10.1007/978-3-031-70365-2_22).

## Usage

The following code demonstrates how to use the `ToCanonizedDirectedAcyclicGraph` class from the dag_gnn module to transform graph data into a DAG format.
This transformation is useful in graph-related tasks such as node classification or graph classification.

```python
from src.dag_gnn import ToCanonizedDirectedAcyclicGraph

k_redundancy = 0  # The hyperparameter that controls the redundancy of the graph
number_of_nodes = ...  # Set this to `None` for graph classification tasks
number_of_layers_for_your_dag_mlp_model = ...  # Must be greater than or equal to 1

dag_transform = ToCanonizedDirectedAcyclicGraph(num_nodes=number_of_nodes,
                                                num_layers=number_of_layers_for_your_dag_mlp_model,
                                                k=k_redundancy)
data = dag_transform(data)

# In the forward function, you extract the edges per layer:
def forward(self, data):
    dag_edge_index = data.dag_edge_index
    dag_layers_mask = data.dag_layers_mask
    dag_edge_attr = data.edge_multiplicities
    dag_readouts = data.dag_readout_at_each_layer
    dag_leaves_at_each_layer = data.dag_leaves_at_each_layer
    dag_x = data.dag_x
    leaves_0 = dag_leaves_at_each_layer[0]

    feature = ...  # Transform original node features `dag_x` to an embedding space

    x = zeros_like(feature)
    x[leaves_0] = feature[leaves_0]

    for i in range(self.num_layers):
        edge_index_i = dag_edge_index[:, dag_layers_mask == i]
        dag_edge_attr_i = dag_edge_attr[dag_layers_mask == i]
        # Additional operations per layer can be performed here
        ...

    x_at_each_layer = []
    for i in range(self.num_layers + 1):
        readout_i = dag_readouts[i]
        x_at_each_layer.append(x[readout_i])


```

## Terms and conditions
When using our code please cite our paper:

```bibtex
@inproceedings{bause2024redundancy,
  title={On the Two Sides of Redundancy in Graph Neural Networks},
  author={Franka Bause, Samir Moustafa, Johannes Langguth, Wilfried N. Gansterer, and Nils M. Kriege},
  booktitle={ECML/PKDD, Lecture Notes in Computer Science 14946},
  year={2024},
}
```


## Installation
Docker installation:
```bash
# 1. Build the docker image:
docker build -t kredundancygnn .
# 2. Run the docker container and attach to bash:
docker run -it --rm --gpus all kredundancygnn /bin/sh
```

Local installation:
```bash
# 1. (Optional) Create new environment for the the project:
conda create -n k_redundancy_gnn python=3.11     # it is needed to install python 3.10
# 2. (Optional) Activate the new environment:
conda activate k_redundancy_gnn
# 3. (Optional) Install cudatoolkit 11.3 and PyTorch dependencies (if you have GPU, otherwise skip this step):
conda install pytorch cudatoolkit=11.3 -c pytorch
# 4. Clone the repository:
git clone https://github.com/SamirMoustafa/k-RedundancyGNNs.git
# 5. Install the dependencies:
cd k-RedundancyGNNs && pip install -r requirements.txt
# 6. Export the repository path to PYTHONPATH *temporarily*:
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Reproducing the results
The results can be reproduced by running the following commands:
```bash
# Synthetic datasets (CSL, EXP)
python task_graph/synthetic/synthetic_case_study_run.py
python task_graph/synthetic/synthetic_run.py --dataset CSL
python task_graph/synthetic/synthetic_run.py --dataset EXP
# Node Classification (Cora, Citeseer, Pubmed, Cornell, Texas, Wisconsin)
python task_node/main_dagmlp.py
# TU datasets (IMDB-BINARY, IMDB-MULTI, ENZYMES, PROTEINS)
python task_graph/tudataset/tu_datasets_run.py
```


## Repository Structure
```tree
.
|-- data
|   |-- CSL (with subdirectories for the data)
|   `-- EXP (with subdirectories for the data)
|-- src
|   |-- __init__.py
|   |-- canonized_dag.py
|   |-- canonized_ntrees.py
|   |-- dag.py
|   |-- dag_gnn.py
|   |-- hash_function.py
|   |-- ntrees.py
|   |-- process_daemon.py
|   `-- tensor_dag.py
|-- task_graph
|   |-- synthetic
|   |   |-- dagmlp_model.py
|   |   |-- dataset
|   |   |   |-- CSL.py
|   |   |   `-- EXP.py
|   |   |-- loader.py
|   |   |-- synthetic_case_study_run.py
|   |   `-- synthetic_run.py
|   `-- tudataset
|       |-- dagmlp_model.py
|       |-- statistics.py
|       `-- tu_datasets_run.py
|-- task_node
|   |-- dagmlp_model.py
|   |-- main_dagmlp.py
|   `-- main_gin.py
|-- test
|   |-- test_canonized_dag.py
|   |-- test_dag.py
|   |-- test_dag_mlp_layer.py
|   `-- test_hash_function.py
|-- requirements.txt
|-- Dockerfile
|-- README.md
`-- utils.py
```


## Contact information
If you have any questions, please contact [Franka Bause](https://dm.cs.univie.ac.at/team/person/112939/).
