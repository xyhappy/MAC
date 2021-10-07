# Multi-structure Graph Classification Method with Attention-based Pooling

This is a PyTorch implementation of MAC algorithm, which learns a graph-level representation for the entire graph. Specifically, the graph pooling operator adopts multiple strategies to evaluate the importance of nodes and update node representations through attention mechanism. Then a hierarchical architecture is designed to capture multiple different substructures of the input graph. Finally, a 2D CNN is used to generate a graph-level representation.

## Requirements

- python3.7
- torch==1.7.0
- dgl==0.6.1

Note:

This code repository is built on [dgl](https://github.com/dmlc/dgl), which is a Python package built for easy implementation of graph neural network model family. Please refer [here](https://docs.dgl.ai/install/index.html) for how to install and utilize the library.

### Datasets

Graph classification benchmarks are publicly available at [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

### Run

To run MAC, just execute the following command for graph classification task:

```
python main.py
```

### Parameter Settings

| Datasets     | batch_size | conv_channel1 | conv_channel2 | dropout_ratio |   lr   | pooling_ratio |
| ------------ | :--------: | :-----------: | :-----------: | :-----------: | :----: | :-----------: |
| PROTEINS     |     64     |      16       |       8       |      0.0      | 0.001  |      0.3      |
| DD           |     64     |       8       |       4       |      0.7      | 0.001  |      0.6      |
| NCI1         |     32     |      32       |      16       |      0.3      | 0.0005 |      0.4      |
| NCI109       |     32     |      16       |       8       |      0.2      | 0.001  |      0.6      |
| Mutagenicity |     32     |       8       |      32       |      0.5      | 0.001  |      0.5      |

Note:

PROTEINS, NCI1, NCI109 and Mutagenicity are running on GeForce RTX 2080 Ti, DD is running on Tesal V100.

