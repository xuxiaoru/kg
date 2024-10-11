感谢你指出这个问题！代码中 `hits` 总是与 `10` 相同，可能是因为在 `hit_at_10` 函数中计算时，误将所有边（包括正例和负例）都用于 `sorted_edges[:10]`，导致正例过于容易出现在前10个预测中。

我们需要修正 `hit_at_10` 的计算逻辑，以便确保只使用正例边的排名来评估 Hit@10，而不是所有边都参与计算。

### 修改 `hit_at_10` 逻辑

正确的逻辑应该是：
1. 只对正例边进行排序（通过其对应的得分），然后检查它们是否出现在前 10 个预测边中。
2. 通过排序后的预测结果查看正例边是否被正确预测为前 10 个。

### 修正代码

```python
def hit_at_10(positive_edges, all_edges, scores):
    # 1. 对所有边按照分数排序
    sorted_edges_with_scores = sorted(zip(all_edges, scores), key=lambda x: x[1], reverse=True)
    
    # 2. 取前10个得分最高的边
    top_10_edges = [edge for edge, score in sorted_edges_with_scores[:10]]
    
    # 3. 统计有多少正例边出现在前10个预测边中
    hits = sum([1 for edge in positive_edges if edge in top_10_edges])
    
    # 计算命中率
    return hits / len(positive_edges)

# 重新计算 Hit@10
hit10 = hit_at_10(positive_edges, all_edges, scores)
print(f"Hit@10: {hit10}")
```

### 主要修改点：
1. **排序并筛选前 10 个预测结果**：在 `hit_at_10` 函数中，我们按照得分对 `all_edges` 进行排序，只取前 10 个边。
2. **比较正例边是否在前 10**：然后我们检查这些正例边是否出现在排序后的前 10 个边中，并统计有多少正例命中前 10。

### 完整代码

```python
import networkx as nx
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

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
    # 1. 对所有边按照分数排序
    sorted_edges_with_scores = sorted(zip(all_edges, scores), key=lambda x: x[1], reverse=True)
    
    # 2. 取前10个得分最高的边
    top_10_edges = [edge for edge, score in sorted_edges_with_scores[:10]]
    
    # 3. 统计有多少正例边出现在前10个预测边中
    hits = sum([1 for edge in positive_edges if edge in top_10_edges])
    
    # 计算命中率
    return hits / len(positive_edges)

# 计算 Hit@10
hit10 = hit_at_10(positive_edges, all_edges, scores)
print(f"Hit@10: {hit10}")
```

### 输出

1. **链接预测的准确率**：输出模型预测边存在的准确性。
2. **Hit@10**：输出正确边出现在前 10 个预测中的命中率。

### 结论

通过修正 `hit_at_10` 函数，你现在可以正确评估模型在前 10 个预测边中的命中情况。这是一个常见的错误，确保我们只用模型预测中的**前 10 个**结果来计算 Hit@10，而不是所有预测边。