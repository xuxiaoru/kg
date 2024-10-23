下面是基于你的要求，完整生成的代码，使用 `TransE` 模型训练，并处理节点 ID 顺序和关系类型重复值的问题。代码流程如下：

1. **处理数据，生成模型所需格式**：将 `nodes` 和 `relationships` 提取出来，生成三元组（head, relation, tail）。
2. **使用 PyKEEN 的 TransE 进行训练**。
3. **保存实体和关系的嵌入到 CSV 文件**。
4. **对新数据生成嵌入，并从已有嵌入集中召回前 3 条**。

### 完整代码实现

```python
import json
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.models import TransE
import torch
import numpy as np

# 1. 解析 JSON 数据
json_data_A = '''
{
    "nodes": [
        {"id": "12-996", "label": "流程", "properties": {"cn_name": "流程A", "desc_cn": "描述A"}},
        {"id": "12-454", "label": "流程", "properties": {"cn_name": "流程B", "desc_cn": "描述B"}},
        {"id": "12-243", "label": "流程", "properties": {"cn_name": "流程C", "desc_cn": "描述C"}},
        {"id": "12-867", "label": "流程", "properties": {"cn_name": "流程D", "desc_cn": "描述D"}}
    ],
    "relationships": [
        {"id": "001-11", "startnode": "12-996", "endnode": "12-454", "type":"friendof"},
        {"id": "001-23", "startnode": "12-996", "endnode": "12-243", "type":"friendof"},
        {"id": "001-12", "startnode": "12-243", "endnode": "12-867", "type":"wideof"}
    ]
}
'''

data = json.loads(json_data_A)

# 2. 处理关系，去重并保持顺序
triples = []
unique_relation_types = list(dict.fromkeys([rel['type'] for rel in data['relationships']]))

for rel in data['relationships']:
    head = rel['startnode']
    relation = rel['type']
    tail = rel['endnode']
    triples.append((head, relation, tail))

# 3. 转换为 PyKEEN 可接受的格式 (三元组)
training_triples = pd.DataFrame(triples, columns=["head", "relation", "tail"])

# 4. 使用 PyKEEN 的 TransE 模型进行训练
result = pipeline(
    training=training_triples,
    model='TransE',
    training_loop='lcwa',
    epochs=100,
)

# 5. 保存模型
model = result.model
model.save_model('transe_model.pkl')

# 6. 提取实体和关系的嵌入
entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
relation_embeddings = model.relation_representations[0](indices=None).detach().cpu().numpy()

# 7. 获取节点和关系ID
entity_ids = list(dict.fromkeys(
    [rel['startnode'] for rel in data['relationships']] + 
    [rel['endnode'] for rel in data['relationships']]
))

# 8. 保存实体和关系嵌入到 CSV
entity_embeddings_df = pd.DataFrame(entity_embeddings, index=entity_ids)
relation_embeddings_df = pd.DataFrame(relation_embeddings, index=unique_relation_types)

entity_embeddings_df.to_csv('entity_embeddings.csv', index=True, header=False)
relation_embeddings_df.to_csv('relation_embeddings.csv', index=True, header=False)

# 9. 生成新数据的嵌入，并进行召回
def get_embedding_for_node(node_id):
    """根据node_id获取对应的嵌入"""
    index = entity_ids.index(node_id)
    return entity_embeddings[index]

def recall_top_k(embedding, k=3):
    """根据嵌入召回前k个最相似的节点"""
    distances = np.linalg.norm(entity_embeddings - embedding, axis=1)
    top_k_indices = np.argsort(distances)[:k]
    return [entity_ids[i] for i in top_k_indices]

# 示例：生成节点 '12-996' 的嵌入，并召回最相似的 3 个节点
new_node_embedding = get_embedding_for_node('12-996')
top_3_recall = recall_top_k(new_node_embedding, k=3)
print("召回的前3个最相似的节点ID：", top_3_recall)
```

### 代码说明

1. **处理数据并生成三元组**：通过 `data['relationships']` 提取三元组 `(head, relation, tail)`，并对关系类型 `type` 去重处理，以避免重复。
2. **TransE 模型训练**：使用 PyKEEN 的 `pipeline` 方法，基于生成的三元组数据训练 TransE 模型，并保存模型文件。
3. **生成和保存嵌入**：提取训练好的实体和关系的嵌入，并将其与对应的节点 ID 和关系类型关联，保存为 CSV 文件。
4. **嵌入召回功能**：提供对新节点的嵌入生成功能，并根据欧几里得距离（`np.linalg.norm`）计算相似度，召回最相似的 3 个节点。

### 处理关系类型重复的问题

在代码的第 2 步中，`unique_relation_types` 列表通过 `dict.fromkeys` 进行去重处理。这样，在生成关系嵌入时，会确保每种关系类型只生成一个对应的嵌入。
