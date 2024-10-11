在这个示例中，我们将使用 **Node2Vec** 算法进行**图嵌入**，然后通过这些嵌入来实现**链接预测**，并且使用 **Hit@10** 作为评估指标。

步骤包括：
1. **图数据准备**：构建图并使用 `Node2Vec` 算法生成节点的嵌入。
2. **链接预测任务**：给定一些已经存在的边（正例），生成一些不存在的边（负例），并利用节点嵌入进行边的预测。
3. **评估**：计算 Hit@10 评估指标。

### 1. 安装必要的依赖

你需要安装 `networkx` 和 `node2vec` 库：

```bash
pip install networkx node2vec gensim
```

### 2. 代码实现

以下是完整的 Python 示例代码，展示如何通过 Node2Vec 进行图嵌入，链接预测，以及使用 Hit@10 评估模型的性能。

```python
import networkx as nx
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score

# 1. 图数据准备
# 创建一个示例图
G = nx.Graph()

# 添加边 (假设这是你原始的图结构，带有一些节点和边)
edges = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
G.add_edges_from(edges)

# 2. 生成节点嵌入 (Node2Vec)
node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=200, workers=4)
model = node2vec.fit(window=5, min_count=1, batch_words=4)

# 获取两个节点的嵌入，并计算他们的余弦相似度
def cosine_similarity(node1, node2):
    vec1 = model.wv[str(node1)]
    vec2 = model.wv[str(node2)]
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return sim

# 3. 准备链接预测数据
# 3.1 创建正例（已存在的边）
positive_edges = list(G.edges())

# 3.2 生成负例（不存在的边）
# 从未连接的节点对中随机选择一些作为负例
non_edges = list(nx.non_edges(G))
negative_edges = np.random.choice(len(non_edges), len(positive_edges), replace=False)
negative_edges = [non_edges[i] for i in negative_edges]

# 将正例和负例合并为一个数据集
all_edges = positive_edges + negative_edges
labels = [1] * len(positive_edges) + [0] * len(negative_edges)

# 4. 计算每对边的相似度得分（使用余弦相似度）
scores = [cosine_similarity(u, v) for u, v in all_edges]

# 5. 评估链接预测性能
# 使用简单的阈值计算预测结果
threshold = 0.5
predictions = [1 if score > threshold else 0 for score in scores]

# 计算 AUC 或准确率
accuracy = accuracy_score(labels, predictions)
print(f"Link prediction accuracy: {accuracy}")

# 6. Hit@10 评估
# 定义 Hit@10 评估函数，查看正确的边是否出现在前 10 个预测中
def hit_at_10(positive_edges, all_edges, scores):
    sorted_edges = [edge for _, edge in sorted(zip(scores, all_edges), reverse=True)]
    hits = 0
    for edge in positive_edges:
        if edge in sorted_edges[:10]:  # 检查是否在前10个预测中
            hits += 1
    return hits / len(positive_edges)

# 计算 Hit@10
hit10 = hit_at_10(positive_edges, all_edges, scores)
print(f"Hit@10: {hit10}")
```

### 代码详解

1. **图数据准备**：
   - 我们构建了一个简单的无权图 `G`，其中包含一些节点和边。

2. **节点嵌入**：
   - 使用 `Node2Vec` 生成节点的嵌入，每个节点的嵌入维度设置为 64，随机游走的长度为 10。
   - `fit()` 方法训练了模型，我们可以使用 `model.wv[node]` 获取节点的嵌入向量。

3. **链接预测任务**：
   - 首先，我们将图中的现有边作为正例。
   - 然后，从图中不存在的边中随机采样一些作为负例。
   - 使用嵌入向量的**余弦相似度**来预测边是否存在。

4. **模型评估**：
   - 我们使用余弦相似度和简单的阈值判断两节点是否连接，并使用**准确率**作为评估指标之一。
   - **Hit@10** 指标用于评估模型在前 10 个预测中是否包含正确的边，即正例是否出现在得分最高的前 10 个边中。

### 输出结果

该代码会输出两个评估结果：

1. **链接预测的准确率**：衡量预测边是否存在的准确性。
2. **Hit@10**：衡量正例边是否出现在前 10 个预测边中的比例。

### 调整与扩展
- 可以进一步调优 `Node2Vec` 的超参数（如 `p` 和 `q`），以探索不同的图搜索策略（DFS 或 BFS 偏好）。
- 可以考虑不同的相似度计算方式（例如欧几里得距离或点积），以替代余弦相似度来进行链接预测。 
- 在大型图中，可以使用更复杂的边采样策略生成负例。