在使用 Node2Vec 生成图嵌入时，本地数据通常是一个**图结构**，由节点和边组成。可以通过 **边列表** 或 **邻接矩阵** 的形式来表示这个图。下面是一些本地数据格式的示例，你可以根据你的数据进行适当的调整。

### 1. 边列表格式
边列表是最常见的数据格式之一，它定义了每一条边的两个端点，也就是哪些节点之间有连接。

#### 示例数据
```plaintext
# 边列表文件 (edges.txt)
1 2
1 3
2 3
3 4
```

这个文件表示了一个简单的无向图：
- 节点 1 与节点 2 相连
- 节点 1 与节点 3 相连
- 节点 2 与节点 3 相连
- 节点 3 与节点 4 相连

#### 读取并构建图
你可以使用 `NetworkX` 读取边列表并构建图，然后将其用于 `Node2Vec`。

```python
import networkx as nx
from node2vec import Node2Vec

# 读取边列表文件并构建无向图
G = nx.read_edgelist('edges.txt', nodetype=int)

# 训练 Node2Vec 模型
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1)

# 获取节点的嵌入
embeddings = model.wv
```

### 2. 邻接矩阵格式
邻接矩阵是一种二维矩阵，用来表示图中节点之间的连接关系。如果两个节点之间有连接，对应矩阵位置的值为 1，否则为 0。

#### 示例数据
```plaintext
# 邻接矩阵 (adj_matrix.txt)
0 1 1 0
1 0 1 0
1 1 0 1
0 0 1 0
```

这个矩阵表示的图与上面的边列表表示的图是相同的。每行每列表示一个节点，行列的交点表示该节点与其他节点的连接：
- 节点 1（第 1 行）与节点 2 和 3 相连
- 节点 2（第 2 行）与节点 1 和 3 相连
- 节点 3（第 3 行）与节点 1、2 和 4 相连
- 节点 4（第 4 行）与节点 3 相连

#### 读取并构建图
你可以将邻接矩阵转换成边列表，或者直接使用图构建工具来生成图。

```python
import numpy as np
import networkx as nx

# 从邻接矩阵文件中读取数据
adj_matrix = np.loadtxt('adj_matrix.txt', dtype=int)

# 构建图对象
G = nx.from_numpy_matrix(adj_matrix)

# 训练 Node2Vec 模型
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1)

# 获取节点的嵌入
embeddings = model.wv
```

### 3. JSON 格式 (带属性的图)
如果你的图中有更多的节点或边属性，可以使用 JSON 格式来存储。

#### 示例数据
```json
{
  "nodes": [
    {"id": 1, "label": "Node1"},
    {"id": 2, "label": "Node2"},
    {"id": 3, "label": "Node3"},
    {"id": 4, "label": "Node4"}
  ],
  "edges": [
    {"source": 1, "target": 2},
    {"source": 1, "target": 3},
    {"source": 2, "target": 3},
    {"source": 3, "target": 4}
  ]
}
```

这个 JSON 文件描述了一个带属性的图，其中 `nodes` 是节点信息，`edges` 是边的连接关系。

#### 读取并构建图
```python
import networkx as nx
import json

# 读取 JSON 文件
with open('graph.json', 'r') as f:
    data = json.load(f)

# 构建无向图
G = nx.Graph()

# 添加节点
for node in data['nodes']:
    G.add_node(node['id'], label=node['label'])

# 添加边
for edge in data['edges']:
    G.add_edge(edge['source'], edge['target'])

# 训练 Node2Vec 模型
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1)

# 获取节点的嵌入
embeddings = model.wv
```

### 总结
- **边列表格式**：适合简单的无向图和有向图，文件内容列出节点的连接关系。
- **邻接矩阵格式**：用于描述较小规模的图，每个位置的值表示节点之间的连接关系。
- **JSON 格式**：适合复杂的图，包含节点和边的额外属性信息。

你可以根据自己数据的实际需求选择合适的格式进行图嵌入生成。
