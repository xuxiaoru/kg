以下是一个完整的示例，展示如何使用 PyKEEN 训练 TransE 模型，包括数据集准备、模型训练、嵌入结果保存以及与原始数据的关联。

### 1. 准备自定义数据集

首先，创建一个包含三元组的文件，例如 `train.txt`，内容如下：

```
0 0 1
0 1 2
1 0 2
```

### 2. 安装 PyKEEN

确保你已经安装了 PyKEEN：

```bash
pip install pykeen
```

### 3. 训练 TransE 模型

以下是训练 TransE 模型的完整示例代码：

```python
import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.models import TransE
from pykeen.training import TrainingLoop
from pykeen.evaluation import Evaluator

# 1. 加载自定义数据集
train_triples = TriplesFactory.from_path('path/to/train.txt')  # 替换为实际路径

# 2. 创建 TransE 模型
model = TransE(
    triples_factory=train_triples,
    embedding_dim=50,  # 嵌入维度
)

# 3. 设置训练器
trainer = TrainingLoop(
    model=model,
    triples_factory=train_triples,
    num_epochs=100,  # 训练轮数
    batch_size=512,  # 批大小
)

# 4. 运行训练
trainer.run()

# 5. 获取实体和关系嵌入
entity_embeddings = model.entity_representations[0].detach().numpy()
relation_embeddings = model.relation_representations[0].detach().numpy()

# 6. 保存嵌入结果
entity_df = pd.DataFrame(entity_embeddings, columns=[f'embed_{i}' for i in range(entity_embeddings.shape[1])])
relation_df = pd.DataFrame(relation_embeddings, columns=[f'embed_{i}' for i in range(relation_embeddings.shape[1])])

# 7. 将嵌入与原始数据关联
entity_df['entity_id'] = list(train_triples.entity_to_id.keys())
relation_df['relation_id'] = list(train_triples.relation_to_id.keys())

# 8. 保存为 CSV 文件
entity_df.to_csv('entity_embeddings.csv', index=False)
relation_df.to_csv('relation_embeddings.csv', index=False)

# 9. 评估模型（可选）
evaluator = Evaluator(model=model, triples_factory=train_triples)
results = evaluator.evaluate()
print(results)
```

### 4. 代码解释

1. **加载数据集**：使用 `TriplesFactory.from_path` 从文件加载训练三元组。
2. **创建模型**：使用 `TransE` 类创建模型。
3. **设置训练器**：配置训练参数，包括训练轮数和批大小。
4. **训练模型**：调用 `trainer.run()` 开始训练。
5. **获取嵌入**：从模型中提取实体和关系的嵌入向量。
6. **保存结果**：将嵌入向量保存到 CSV 文件中，并将实体和关系的 ID 添加到 DataFrame。
7. **评估模型**：可选地评估模型性能并打印结果。

### 注意事项

- 请确保数据文件路径正确。
- 根据你的数据集调整参数（例如，嵌入维度和训练轮数）。