要开始使用 FastAPI，首先需要进行安装，并了解如何快速构建一个 API。以下是 FastAPI 的安装步骤和一个简单的使用示例：

### 一、安装 FastAPI 和 Uvicorn

FastAPI 依赖于 ASGI 服务器来运行，因此通常需要安装 `uvicorn`，这是一个支持异步 Python web 框架的轻量级 ASGI 服务器。

#### 1. 安装 FastAPI 和 Uvicorn
可以通过 `pip` 来安装：

```bash
pip install fastapi
pip install uvicorn[standard]
```

- `fastapi` 是框架本身，提供了开发 API 的主要功能。
- `uvicorn` 是用于运行 FastAPI 应用的 ASGI 服务器。

### 二、编写第一个 FastAPI 应用

以下是一个简单的 FastAPI 应用程序示例：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}
```

#### 代码说明：
- `FastAPI()`：实例化 FastAPI 应用。
- `@app.get("/")`：使用装饰器定义一个 GET 请求的路径操作函数。当访问根路径 `/` 时，该函数会被调用。
- `@app.get("/items/{item_id}")`：通过路径参数 `item_id` 和可选查询参数 `q` 处理 `/items/{item_id}` 路径的请求。

### 三、运行 FastAPI 应用

1. 将上面的代码保存到一个 Python 文件中，比如 `main.py`。
2. 使用 `uvicorn` 启动 FastAPI 应用：
   
   ```bash
   uvicorn main:app --reload
   ```

   解释：
   - `main:app`：`main` 指代 Python 文件名，`app` 是在文件中创建的 FastAPI 实例。
   - `--reload`：启用自动重新加载。当代码修改时，服务器会自动重新启动，便于开发调试。

3. 启动后，服务器会在默认的 `http://127.0.0.1:8000` 运行。打开浏览器访问：

   - [http://127.0.0.1:8000](http://127.0.0.1:8000)：返回 `{"message": "Hello, FastAPI"}`。
   - [http://127.0.0.1:8000/items/42?q=sample](http://127.0.0.1:8000/items/42?q=sample)：返回 `{"item_id": 42, "query": "sample"}`。

### 四、交互式文档

FastAPI 自动生成文档，方便测试 API：

- 访问 Swagger UI：`http://127.0.0.1:8000/docs`
- 访问 ReDoc 文档：`http://127.0.0.1:8000/redoc`

Swagger UI 提供交互式文档，用户可以直接在浏览器中调用 API 并查看结果。ReDoc 则展示更详细的文档说明。

### 五、更多功能示例

#### 1. 添加 POST 请求
FastAPI 支持多种 HTTP 请求方法，例如 POST。可以使用 `@app.post` 来定义 POST 请求的路径操作：

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

@app.post("/items/")
def create_item(item: Item):
    return {"item_name": item.name, "item_price": item.price}
```

- `BaseModel` 是 FastAPI 用于数据验证和解析的 Pydantic 模型。
- 当发送一个 POST 请求到 `/items/` 路径时，FastAPI 会根据定义的 `Item` 模型自动验证请求体的数据格式。

运行服务器后，你可以通过 Swagger UI 或其他工具向 `/items/` 发送 POST 请求，带上 JSON 数据，例如：
```json
{
    "name": "Laptop",
    "price": 1200.50,
    "is_offer": false
}
```

#### 2. 异步支持
FastAPI 完全支持 Python 的异步操作。通过使用 `async def`，可以异步处理请求：

```python
@app.get("/async-items/")
async def read_async_items():
    return {"message": "This is an asynchronous request"}
```

### 六、快速部署

1. **通过 Docker 部署**：
   FastAPI 和 Uvicorn 可以很容易通过 Docker 部署。创建一个 `Dockerfile`：
   
   ```Dockerfile
   FROM python:3.9
   WORKDIR /app
   COPY . /app
   RUN pip install fastapi uvicorn
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
   ```

2. **通过云服务（如 AWS、Google Cloud、Heroku）部署**：
   FastAPI 应用可以轻松部署在支持 Python 的任何云平台上。大多数云平台都支持 ASGI 框架和服务，因此你可以按照其文档快速部署。

### 七、总结

1. **安装步骤**：
   - 使用 `pip` 安装 FastAPI 和 Uvicorn。
2. **运行步骤**：
   - 使用 `uvicorn main:app --reload` 启动开发服务器。
3. **交互式文档**：
   - Swagger UI 和 ReDoc 提供方便的 API 文档和调试工具。
4. **异步、数据验证、自动文档**：
   - FastAPI 提供了异步操作支持、自动数据验证和自动生成 API 文档的功能，非常适合构建现代化的 Web API。

通过 FastAPI 的灵活性和高性能，开发者能够快速搭建从简单到复杂的 API，适合小型项目到大规模服务的开发与部署。
