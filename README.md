# Study Assistant

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Web%20Framework-green)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)

Study Assistant 是一个基于 **RAG架构** 构建的练习力扣Hot100本地算法学习辅助系统。

用户可以输入一道新题的题意描述， 系统通过向量检索在 力扣Hot100 中匹配相似题目，再结合本地大语言模型生成知识点判断、原因分析与练习建议， 帮助用户把陌生题目快速映射到高频经典题型。

同时，系统也支持题目详情查询、分类推荐与错题统计，用于构建更完整的算法练习与复习流程。

整个系统采用 **本地部署** 的方式运行，结合向量检索、RAG 推理和 Web 界面，实现一个完整的算法学习辅助工具。

---

# 系统演示


<img width="1197" height="716" alt="截屏2026-03-18 03 53 08" src="https://github.com/user-attachments/assets/b0a5d712-f259-4108-b3b5-c4e4e54c3694" />
页面集成了多个功能模块：

* RAG 语义搜索
* 题目详情查询
* 分类推荐练习
* 错题记录与统计

用户可以通过网页界面直接进行算法学习辅助。



# 系统架构

采用典型的 **RAG 应用架构**：

```text
用户输入题意
      │
      ▼
Embedding 向量化
      │
      ▼
FAISS 向量检索
      │
      ▼
候选题目 Top-K
      │
      ▼
LLM 生成分析（RAG）
```

核心组件：

* **LangChain**：构建 RAG 推理流程
* **FAISS**：向量数据库，用于语义检索
* **Ollama**：运行本地大语言模型
* **FastAPI**：提供 API 服务
* **Jinja2**：Web 页面模板



# 数据集

当前系统使用 **LeetCode Hot100 题库** 作为检索数据源。

目前数据集包含两种难度的题目类别：

* Easy
* Medium

每道题目包含以下信息：

* 题号
* 标题
* 难度
* 题目描述
* 解题思路
* Python 示例代码
* 算法类别

未来可以扩展：

* Hard 难度题目
* 更大规模算法题库
* 多语言解法（Python / Java / C++）


# 系统功能

当前版本支持以下能力：



# RAG 题意分析

用户可以输入算法题目的自然语言描述，例如：

```
给定数组，找到三个数之和为0
```

系统会进行语义检索，并返回最相似的题目，例如：

```
LeetCode 15 三数之和
```

同时结合本地大模型生成：

* 知识点分析
* 解题思路
* 推荐练习

<img width="1189" height="780" alt="截屏2026-03-18 03 54 55" src="https://github.com/user-attachments/assets/affad273-2b30-485c-b2ec-e8edd0fb1462" />
<img width="1185" height="781" alt="截屏2026-03-18 03 55 07" src="https://github.com/user-attachments/assets/be6d9c1a-39c3-4057-964f-07e5027925dc" />




# 题目详情查询

用户可以查询指定题目的详细信息，包括：

* 题目标题
* 难度
* 类别
* 题目描述
* 解题思路
* Python 示例代码
<img width="1202" height="725" alt="截屏2026-03-18 03 56 19" src="https://github.com/user-attachments/assets/5853fcc6-6644-4972-8525-57a04a0f51e8" />





# 分类推荐

用户可以根据 **算法类别** 和 **难度** 进行题目推荐，例如：

* 双指针
* 哈希
* 滑动窗口

系统会返回对应类别的题目列表，用于定向练习。

<img width="1194" height="740" alt="截屏2026-03-18 03 56 32" src="https://github.com/user-attachments/assets/9d99ac0f-4bbb-49b5-acf7-56636cbc3262" />


# 错题管理

系统支持记录用户练习过程中出现错误的题目。

功能包括：

* 加入错题本
* 查看错题列表
* 错题类别统计

帮助用户建立自己的 **算法学习记录系统**。

<img width="1198" height="775" alt="截屏2026-03-18 03 57 22" src="https://github.com/user-attachments/assets/e4bb06c9-c2f2-4f8b-a7da-3b478b314107" />



# 项目结构

```
Study_assistant
│
├── api
│   └── app.py                # FastAPI 接口
│
├── core
│   ├── loader.py             # 题库加载
│   ├── parser.py             # Markdown 解析
│   ├── vector_store.py       # FAISS 向量数据库
│   ├── rag.py                # RAG 推理逻辑
│   └── wrong_book.py         # 错题管理
│
├── services
│   └── assistant_service.py  # 业务逻辑封装
│
├── web
│   └── templates
│       └── index.html        # Web 页面
│
├── data
│   └── processed             # 处理后的题库数据
│
├── main.py                   # CLI 入口
└── README.md
```

---

# 运行方式

## 1 创建虚拟环境 & 安装依赖

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## 2 启动本地模型

确保已安装 **Ollama**：

```
ollama run llama3
```

或提前下载模型：

```
ollama pull llama3
```


## 3 启动服务

```
uvicorn api.app:app --reload
```


## 4 打开网页

浏览器访问：

```
http://127.0.0.1:8000
```

---

## 5 API 文档

FastAPI 自动生成接口文档：

```
http://127.0.0.1:8000/docs
```

---

# 技术栈

* Python 3.10
* FastAPI
* LangChain
* FAISS
* Ollama
* Jinja2

---

# 后续规划

未来可以扩展：

* 更大规模算法题库
* 支持 Hard 难度题目
* Agent 自动学习助手
* 更智能的学习路径推荐

---

# 项目目标

本项目旨在探索 **RAG 技术在算法学习辅助中的应用**。

通过结合：

* 向量检索
* 本地大语言模型
* Web 应用

构建一个可扩展的 **算法学习助手系统**。

---
