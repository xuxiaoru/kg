要将 Cora 数据集的 `cora.content` 和 `cora.cites` 文件转换为三元组（subject, predicate, object），我们需要先了解这两个文件的内容：

- **cora.content**: 该文件包含了每个论文的特征和标签，通常以制表符分隔。每一行代表一篇论文，其中包含论文的 ID、特征向量和标签。
- **cora.cites**: 该文件包含了论文之间的引用关系，以制表符分隔。每一行表示一个引用关系，格式为 `source_id target_id`，表示 `source_id` 引用 `target_id`。

下面是将这些文件转换为三元组格式的 Python 代码示例：

```python
import pandas as pd

# 读取 cora.content 文件
def load_cora_content(file_path):
    # 读取文件，并指定分隔符
    content_df = pd.read_csv(file_path, sep='\t', header=None)
    content_df.columns = ['paper_id'] + [f'feature_{i}' for i in range(content_df.shape[1] - 2)] + ['label']
    return content_df

# 读取 cora.cites 文件
def load_cora_cites(file_path):
    cites_df = pd.read_csv(file_path, sep='\t', header=None)
    cites_df.columns = ['source_id', 'target_id']
    return cites_df

# 转换为三元组
def convert_to_triples(content_df, cites_df):
    triples = []

    # 从 cora.content 生成三元组
    for _, row in content_df.iterrows():
        paper_id = row['paper_id']
        label = row['label']
        # subject 是论文ID, predicate 是 'has_label', object 是标签
        triples.append((paper_id, 'has_label', label))

    # 从 cora.cites 生成三元组
    for _, row in cites_df.iterrows():
        source_id = row['source_id']
        target_id = row['target_id']
        # subject 是源论文ID, predicate 是 'cites', object 是目标论文ID
        triples.append((source_id, 'cites', target_id))

    return triples

# 主程序
def main():
    # 文件路径
    content_file_path = 'cora.content'
    cites_file_path = 'cora.cites'

    # 加载数据
    content_df = load_cora_content(content_file_path)
    cites_df = load_cora_cites(cites_file_path)

    # 转换为三元组
    triples = convert_to_triples(content_df, cites_df)

    # 输出结果
    for triple in triples:
        print(triple)

if __name__ == '__main__':
    main()
```

### 代码解释：
1. **加载数据**：
   - `load_cora_content` 函数读取 `cora.content` 文件，并为每一列设置合适的列名。
   - `load_cora_cites` 函数读取 `cora.cites` 文件。

2. **生成三元组**：
   - 在 `convert_to_triples` 函数中，遍历 `content_df`，为每个论文生成 `(paper_id, 'has_label', label)` 格式的三元组。
   - 遍历 `cites_df`，为每个引用关系生成 `(source_id, 'cites', target_id)` 格式的三元组。

3. **输出结果**：
   - 在主程序中，调用上述函数并打印生成的三元组。

### 注意事项：
- 请确保在运行代码之前安装了 `pandas` 库，可以使用 `pip install pandas` 安装。
- 更新 `content_file_path` 和 `cites_file_path` 变量以指向你本地的 Cora 数据集文件路径。