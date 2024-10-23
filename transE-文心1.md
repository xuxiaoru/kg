要完成这个任务，我们需要先处理JSON数据，使其适合TransE模型训练的格式，然后使用PyKEEN库进行训练，并生成嵌入向量。以下是详细步骤：

1. 处理数据，生成模型需要的数据格式
PyKEEN 需要数据以三元组（head, relation, tail）的形式提供。首先，我们需要解析 JSON 数据，并将其转换为三元组列表。

python
import json  
  
# 解析 JSON 数据  
json_data = json.loads(json_data_A)  
nodes = json_data['nodes']  
relationships = json_data['relationships']  
  
# 创建三元组列表  
triples = []  
for rel in relationships:  
    startnode = rel['startnode']  
    endnode = rel['endnode']  
    relation = rel['type']  
    triples.append((startnode, relation, endnode))  
  
# 输出三元组列表，供检查  
print(triples)
2. 使用 PyKEEN 的 TransE 训练，并保存模型
安装 PyKEEN：

bash
pip install pykeen
使用 PyKEEN 进行训练并保存模型：

python
from pykeen.pipeline import pipeline  
from pykeen.models import TransE  
from pykeen.datasets import TriplesFactory  
from pykeen.tracking import DeviceConfig  
  
# 将三元组转换为 PyKEEN 的 TriplesFactory  
tf = TriplesFactory.from_triples(triples, entity_labels={node['id']: node['properties']['cn_name'] for node in nodes})  
  
# 使用 PyKEEN 管道进行训练  
result = pipeline(  
    training=tf,  
    model=TransE,  
    model_kwargs=dict(embedding_dim=50),  # 设置嵌入维度  
    training_kwargs=dict(  
        num_epochs=100,  # 设置训练轮数  
        batch_size=32,  # 设置批次大小  
    ),  
    optimizer_kwargs=dict(lr=0.01),  # 设置学习率  
    device=DeviceConfig.CPU,  # 设置为 CPU，如果你有 GPU 可以设置为 DeviceConfig.CUDA  
)  
  
# 保存模型  
model = result.model  
model.save_to_directory('transE_model')
3. 生成图嵌入向量集，与节点和关系ID关联，保存 CSV
获取嵌入向量并保存到 CSV 文件中：

python
import pandas as pd  
import numpy as np  
  
# 获取实体嵌入  
entity_embeddings = model.entity_embeddings.detach().cpu().numpy()  
entity_ids = tf.entity_ids  
  
# 获取关系嵌入  
relation_embeddings = model.relation_embeddings.detach().cpu().numpy()  
relation_ids = tf.relation_ids  
  
# 创建 DataFrame 保存实体嵌入  
entity_df = pd.DataFrame(entity_embeddings, columns=[f'embedding_{i}' for i in range(entity_embeddings.shape[1])])  
entity_df['entity_id'] = entity_ids  
  
# 创建 DataFrame 保存关系嵌入  
relation_df = pd.DataFrame(relation_embeddings, columns=[f'embedding_{i}' for i in range(relation_embeddings.shape[1])])  
relation_df['relation_id'] = relation_ids  
  
# 保存为 CSV 文件  
entity_df.to_csv('entity_embeddings.csv', index=False)  
relation_df.to_csv('relation_embeddings.csv', index=False)  
  
# 如果需要关联节点名称，可以使用以下代码  
node_names = {node['id']: node['properties']['cn_name'] for node in nodes}  
entity_df['cn_name'] = entity_df['entity_id'].map(node_names)  
entity_df.to_csv('entity_embeddings_with_names.csv', index=False)
现在，你应该会在当前目录下看到 entity_embeddings.csv 和 relation_embeddings.csv 文件，其中包含了节点和关系的嵌入向量。entity_embeddings_with_names.csv 文件还包含了节点的中文名称。

请确保在实际执行时，根据具体情况调整路径和参数。
