以下是一个简化版的 **EGES** 算法的代码示例，展示了如何利用图的结构信息和节点的附加信息（Side Information）来进行节点嵌入。该示例使用 `pytorch` 实现，并引入了**注意力机制**来为附加信息分配不同的权重。

假设我们有一个图，并且每个节点具有一些附加信息，如类别、标签等。我们会使用随机游走获取节点的上下文，并结合附加信息进行嵌入学习。

### 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 假设图结构和附加信息数据如下
node_features = {
    '1': [0.5, 1.2],
    '2': [0.1, 1.8],
    '3': [0.9, 1.4],
    '4': [1.5, 0.7],
    '5': [1.0, 0.9],
}

side_information = {
    '1': [0, 1],  # 类别信息，例如节点1的类别为[0, 1]
    '2': [1, 0],
    '3': [0, 1],
    '4': [1, 1],
    '5': [0, 0],
}

# 定义超参数
embedding_dim = 4  # 嵌入维度
side_info_dim = 2  # 附加信息维度
hidden_dim = 8     # 隐藏层大小
lr = 0.01          # 学习率
epochs = 1000      # 训练轮数

# 定义随机游走获取的节点上下文数据
random_walks = [
    ['1', '2', '3'],  # 节点1，2，3是彼此的邻居
    ['2', '3', '4'],  # 节点2，3，4是彼此的邻居
    ['3', '4', '5'],  # 节点3，4，5是彼此的邻居
    ['4', '5', '1']   # 节点4，5，1是彼此的邻居
]

# 定义模型，包含节点嵌入和附加信息嵌入
class EGES(nn.Module):
    def __init__(self, num_nodes, embedding_dim, side_info_dim):
        super(EGES, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)  # 节点嵌入矩阵
        self.side_info_embeddings = nn.Embedding(num_nodes, side_info_dim)  # 附加信息嵌入矩阵
        self.attention_weights = nn.Linear(side_info_dim, 1)  # 注意力权重计算
        
    def forward(self, node_idx):
        node_embed = self.node_embeddings(node_idx)  # 节点嵌入
        side_embed = self.side_info_embeddings(node_idx)  # 附加信息嵌入
        
        # 计算注意力权重
        attention_scores = self.attention_weights(side_embed)
        attention_scores = torch.softmax(attention_scores, dim=0)  # 归一化
        
        # 加权组合节点嵌入和附加信息
        final_embedding = attention_scores * node_embed
        
        return final_embedding

# 初始化模型
num_nodes = len(node_features)
model = EGES(num_nodes=num_nodes, embedding_dim=embedding_dim, side_info_dim=side_info_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练过程
for epoch in range(epochs):
    total_loss = 0
    
    for walk in random_walks:
        context_node_idx = torch.tensor([int(node)-1 for node in walk], dtype=torch.long)  # 随机游走中的节点索引
        
        # 前向传播，计算嵌入
        embeddings = model(context_node_idx)
        
        # 这里使用简单的目标函数，真实任务中会根据具体的上下文进行更复杂的计算
        target_node_idx = context_node_idx[1]  # 目标节点
        loss = criterion(embeddings[0], target_node_idx)  # 简单的预测目标
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss}")

# 打印节点的最终嵌入
node_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
with torch.no_grad():
    final_embeddings = model(node_indices)
    print(f"Final node embeddings: \n{final_embeddings}")
```

### 代码说明：
1. **数据结构**：
   - `node_features`：表示每个节点的特征向量。
   - `side_information`：每个节点的附加信息（例如类别、标签等）。
   - `random_walks`：通过随机游走生成的节点序列，模拟邻居关系。

2. **模型结构**：
   - **节点嵌入矩阵**：用来学习节点本身的嵌入。
   - **附加信息嵌入矩阵**：用来嵌入节点的附加信息。
   - **注意力机制**：通过线性变换来计算附加信息的注意力权重，决定附加信息在最终嵌入中的影响。

3. **训练**：
   - 模型通过对随机游走生成的上下文节点进行训练，目标是使得相邻节点的嵌入更加接近。
   - 注意力机制根据附加信息分配权重，从而结合节点的特征和其附加信息生成最终嵌入。

4. **输出**：
   - 最终的嵌入向量表示了节点结合结构信息和附加信息后的低维表示。

### 总结：
EGES 通过注意力机制为每个节点的附加信息分配权重，灵活地整合了图结构信息和附加信息。这个代码展示了如何实现一个简单版本的 EGES，实际任务中可能需要根据具体应用调整损失函数和随机游走策略。