`ged4py` 提供了多种计算图编辑距离的算法。以下是一些常见算法的具体示例：

### 1. **动态规划算法**
这种算法用于精确计算小型图的编辑距离。虽然它的计算复杂度较高，但适用于节点和边数量较少的情况。

#### 示例代码
```python
import ged4py
import networkx as nx

# 创建图
G1 = nx.Graph()
G1.add_edges_from([(1, 2), (2, 3)])
G2 = nx.Graph()
G2.add_edges_from([(1, 2), (1, 3)])

# 转换为 ged4py 图
ged_graph1 = ged4py.Graph.from_networkx(G1)
ged_graph2 = ged4py.Graph.from_networkx(G2)

# 计算图编辑距离
distance = ged4py.graph_edit_distance(ged_graph1, ged_graph2, algorithm="DP")
print(f"Dynamic Programming Graph Edit Distance: {distance}")
```

### 2. **贪心算法**
贪心算法通过逐步选择最优解来估计图的编辑距离，适合较大图的快速计算。

#### 示例代码
```python
# 继续使用之前创建的图 G1 和 G2

# 计算图编辑距离
distance = ged4py.graph_edit_distance(ged_graph1, ged_graph2, algorithm="Greedy")
print(f"Greedy Algorithm Graph Edit Distance: {distance}")
```

### 3. **模拟退火算法**
模拟退火是一种近似算法，通过随机搜索来寻找图编辑距离的近似值。

#### 示例代码
```python
# 继续使用之前创建的图 G1 和 G2

# 计算图编辑距离
distance = ged4py.graph_edit_distance(ged_graph1, ged_graph2, algorithm="SimulatedAnnealing")
print(f"Simulated Annealing Graph Edit Distance: {distance}")
```

### 4. **下界计算**
下界算法可以快速计算两个图的最小可能编辑距离，用于剪枝。

#### 示例代码
```python
# 继续使用之前创建的图 G1 和 G2

# 计算图的下界
lower_bound = ged4py.graph_edit_distance(ged_graph1, ged_graph2, algorithm="LowerBound")
print(f"Lower Bound Graph Edit Distance: {lower_bound}")
```

### 总结
这些示例展示了如何使用 `ged4py` 的不同算法计算图编辑距离。根据图的规模和复杂性，选择合适的算法可以显著影响计算效率和结果的准确性。每种算法都有其适用的场景和性能特点。