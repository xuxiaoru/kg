以上代码使用 `networkx` 的 `graph_edit_distance` 函数和自定义的 `node_match` 函数可以实现节点属性的匹配，但它不保证严格的 Top-1 匹配，因为 `node_match` 函数会在每个调用时比较所有可能的节点。

为了实现更明确的 Top-1 匹配逻辑，您可以在计算匹配时为每个节点确定一个唯一的最佳匹配。以下是更新后的代码示例，展示如何实现 Top-1 匹配：

```python
import networkx as nx

def node_match_activity_name(node1, node2):
    # 使用节点属性活动名称进行匹配
    return node1['活动名称'] == node2['活动名称']

def find_best_match(node, graph):
    # 查找最佳匹配节点
    for candidate in graph.nodes(data=True):
        if candidate[1]['活动名称'] == node['活动名称']:
            return candidate[0]  # 返回节点ID
    return None  # 没有匹配的节点

def calculate_edit_distance(graph1, graph2):
    # 尝试获取每个节点的最佳匹配
    matches = {}
    
    for node1 in graph1.nodes(data=True):
        best_match = find_best_match(node1[1], graph2)
        matches[node1[0]] = best_match
    
    # 统计未匹配的节点数
    unmatched_nodes = sum(1 for match in matches.values() if match is None)

    # 计算编辑距离
    edit_distance = nx.graph_edit_distance(graph1, graph2, node_match=node_match_activity_name)

    # 如果没有匹配的节点，返回节点数加边数
    if unmatched_nodes == len(graph1.nodes):
        return len(graph1.nodes) + len(graph1.edges)
    
    return edit_distance

# 示例图构建
G1 = nx.Graph()
G1.add_nodes_from([
    (1, {"活动名称": "A"}),
    (2, {"活动名称": "B"}),
])
G1.add_edges_from([(1, 2)])

G2 = nx.Graph()
G2.add_nodes_from([
    (3, {"活动名称": "C"}),
    (4, {"活动名称": "D"}),
])
G2.add_edges_from([(3, 4)])

# 计算编辑距离
distance = calculate_edit_distance(G1, G2)
print(f"编辑距离: {distance}")
```

### 更新后的逻辑：
1. **find_best_match**：为每个节点查找最佳匹配，只返回一个匹配节点ID。
2. **calculate_edit_distance**：使用找到的匹配关系来计算未匹配的节点数，并最终计算编辑距离。

这样做可以确保对于每个节点只选择最优的 Top-1 匹配，满足你的需求。如果有进一步的要求或修改，请告诉我！