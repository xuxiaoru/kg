为了使用TransE模型对给定的图谱数据集进行图嵌入，并保存节点和关系的嵌入向量，我们需要遵循以下步骤。这里，我们将使用pykeen库，它是一个流行的Python库，用于实现各种知识图谱嵌入模型。

首先，我们需要安装pykeen库（如果尚未安装）：

bash
pip install pykeen
接下来，我们将按照您的要求逐步完成流程：

1. 处理数据，生成模型需要的数据格式
我们需要将JSON数据转换为pykeen可以接受的格式，即三元组（头实体，关系，尾实体）的列表。

python
import json  
from pykeen.triples import TriplesFactory  
  
# 解析JSON数据  
data = json.loads(json_data_A)  
nodes = data['nodes']  
relationships = data['relationships']  
  
# 创建实体到索引的映射  
entity_to_index = {node['id']: i for i, node in enumerate(nodes)}  
index_to_entity = {i: node['id'] for i, node in enumerate(nodes)}  
  
# 创建关系到索引的映射（如果需要的话，对于TransE来说通常不需要，但我们可以保留它以便后续使用）  
relation_to_index = {rel['type']: i for i, rel in enumerate(set(r['type'] for r in relationships))}  
index_to_relation = {i: rel_type for rel_type, i in relation_to_index.items()}  
  
# 生成三元组  
triples = []  
for rel in relationships:  
    startnode_id = rel['startnode']  
    endnode_id = rel['endnode']  
    rel_type = rel['type']  
    startnode_index = entity_to_index[startnode_id]  
    endnode_index = entity_to_index[endnode_id]  
    # 注意：对于TransE，关系通常不需要转换为索引，因为模型直接处理字符串关系类型（除非在特定配置下）  
    # 但为了与pykeen的API兼容，我们可以保留一个“关系索引”的概念（尽管在这里它不会改变）  
    # rel_index = relation_to_index[rel_type]  # 这行在TransE中通常不需要  
    triples.append((startnode_index, rel_type, endnode_index))  
  
# 创建TriplesFactory  
tf = TriplesFactory(triples=triples, entity_to_id=entity_to_index, relation_to_id=None)  # 对于TransE，relation_to_id为None
2. 使用pykeen的TransE训练，并保存模型
python
from pykeen.models import TransE  
from pykeen.training import Trainer  
from pykeen.pipeline import pipeline  
  
# 使用pipeline进行简化（也可以手动创建模型、训练器等）  
result = pipeline(  
    training=tf,  
    model=TransE,  
    model_kwargs=dict(embedding_dim=50),  # 设置嵌入维度  
    training_kwargs=dict(  
        num_epochs=100,  # 设置训练轮数  
        batch_size=32,   # 设置批处理大小  
    ),  
    optimizer_kwargs=dict(lr=0.01),  # 设置学习率  
    negative_sampling_kwargs=dict(num_negatives_per_positive=10),  # 设置负采样比例  
)  
  
# 保存模型  
model = result.model  
model.save_to_directory('transE_model')
3. 生成图嵌入向量集，与节点和关系ID关联，保存csv
注意：对于TransE，我们通常只获取节点（实体）的嵌入向量，因为关系在模型中通常被处理为参数或映射，而不是像节点那样有独立的嵌入向量（尽管在某些扩展模型中，关系也可能有嵌入）。

python
import pandas as pd  
  
# 获取节点嵌入  
entity_embeddings = model.entity_embeddings.detach().cpu().numpy()  
  
# 创建DataFrame保存节点嵌入  
node_df = pd.DataFrame(entity_embeddings, columns=[f'embedding_{i}' for i in range(entity_embeddings.shape[1])])  
node_df['node_id'] = [index_to_entity[i] for i in range(len(index_to_entity))]  
  
# 保存为CSV文件  
node_df.to_csv('node_embeddings.csv', index=False)  
  
# 注意：对于关系嵌入，TransE模型不直接提供每个关系的独立嵌入向量。  
# 如果您需要关系嵌入的某种表示，您可能需要考虑使用其他模型（如DistMult、ComplEx等），  
# 或者从TransE模型的参数中导出关系相关的权重（但这通常不是标准的做法）。  
  
# 由于TransE不直接提供关系嵌入，以下代码仅作为示例，说明如何（如果可能的话）保存关系嵌入，  
# 但实际上对于TransE来说，这部分是不适用的。  
# relation_embeddings = ...  # 假设我们有某种方式获取关系嵌入（对于TransE，这通常不是直接的）  
# relation_df = pd.DataFrame(relation_embeddings, columns=[f'embedding_{i}' for i in range(relation_embeddings.shape[1])])  
# relation_df['relation_type'] = list(index_to_relation.values())  
# relation_df.to_csv('relation_embeddings.csv', index=False)  # 这行代码对于TransE来说是不需要的
请确保在运行上述代码之前，您已经正确安装了pykeen库，并且您的Python环境中没有其他冲突。此外，由于pykeen的API可能会随着版本更新而变化，因此请根据您安装的pykeen版本调整代码。

最后，请注意，对于关系嵌入的部分，由于TransE模型不直接提供每个关系的独立嵌入向量，因此相关代码已被注释掉。如果您需要关系嵌入，请考虑使用其他模型或方法。
