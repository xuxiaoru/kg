以下是一个完整的示例，演示如何准备自定义数据集、训练 TransE 模型、保存嵌入结果，并将嵌入与原始数据关联。

### 1. 数据集准备

假设你有一个自定义数据集，包含三元组，每行格式为 `头实体ID 关系ID 尾实体ID`。将其保存为 `train.txt` 文件。

示例 `train.txt` 文件内容：
```
0 0 1
0 1 2
1 0 2
```

### 2. 训练 TransE 模型

以下代码将加载数据集，训练 TransE 模型，并保存嵌入结果：

```python
import pandas as pd
from pykeen.datasets import TriplesFactory
from pykeen.models import TransE
from pykeen.training import TrainingLoop
from pykeen.evaluation import Evaluator

# 加载自定义数据集
train_triples = TriplesFactory.from_path('path/to/train.txt')  # 替换为实际路径

# 创建 TransE 模型
model = TransE(
    triples_factory=train_triples,
    embedding_dim=50,  # 嵌入维度
)

# 设置训练器
trainer = TrainingLoop(
    model=model,
    triples_factory=train_triples,
    num_epochs=100,  # 训练轮数
    batch_size=512,  # 批大小
)

# 运行训练
trainer.run()

# 获取实体和关系嵌入
entity_embeddings = model.entity_representations[0].detach().numpy()
relation_embeddings = model.relation_representations[0].detach().numpy()

# 保存嵌入结果
entity_df = pd.DataFrame(entity_embeddings, columns=[f'embed_{i}' for i in range(entity_embeddings.shape[1])])
relation_df = pd.DataFrame(relation_embeddings, columns=[f'embed_{i}' for i in range(relation_embeddings.shape[1])])

# 将嵌入与原始数据关联
entity_df['entity_id'] = train_triples.entity_to_id.keys()
relation_df['relation_id'] = train_triples.relation_to_id.keys()

# 保存为 CSV 文件
entity_df.to_csv('entity_embeddings.csv', index=False)
relation_df.to_csv('relation_embeddings.csv', index=False)

# 评估模型（可选）
evaluator = Evaluator(model=model, triples_factory=train_triples)
results = evaluator.evaluate()
print(results)
```

### 3. 关联原始数据

在保存嵌入结果时，代码将实体和关系的 ID 添加到 DataFrame 中，便于后续分析和对照。最后，嵌入结果会保存为 `entity_embeddings.csv` 和 `relation_embeddings.csv` 文件。

### 4. 注意事项

- 确保三元组文件路径正确，并根据需要调整模型参数。
- 嵌入结果中的列名（`embed_0`, `embed_1`, 等）可根据需要修改。
- 可选的评估部分可以根据实际情况进行调整或省略。