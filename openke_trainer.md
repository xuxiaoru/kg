在 OpenKE 中，`Trainer` 类没有 `get_entities()` 和 `get_relations()` 方法。要获取实体和关系的嵌入向量，可以使用 `get_entities` 和 `get_relations` 函数，这些函数通常在模型训练完成后可用。请参考以下示例代码：

### 训练模型并获取嵌入向量

```python
from openke.configuration import Trainer, Config

# 配置训练参数
config = Config()
config.set_in_path('data/train.txt')  # 训练数据路径
config.set_test_path('data/test.txt')  # 测试数据路径
config.set_log_dir('log/')
config.set_create_db(True)
config.set_model('TransE')
config.set_dimension(100)  # 嵌入维度
config.set_epoch(100)  # 训练轮数
config.set_batch_size(128)
config.set_lr(0.01)  # 学习率
config.set_neg_rate(1)  # 负样本数量

# 初始化并运行训练
trainer = Trainer(config)
trainer.run()

# 获取嵌入向量
entities = trainer.get_entities()
relations = trainer.get_relations()

# 输出嵌入向量
for idx, embedding in enumerate(entities):
    print(f"Entity ID: {idx}, Embedding: {embedding}")

for idx, embedding in enumerate(relations):
    print(f"Relation ID: {idx}, Embedding: {embedding}")
```

### 检查函数
如果仍然遇到问题，请确认你的 OpenKE 版本，因为不同版本的实现可能有所不同。在最新版本中，通常可以通过训练后调用这些方法来获取嵌入。如果你的版本不支持这些方法，可以查阅相关文档或示例代码。