# About

This repository is a simple implementation of RAG. My goal is just for learning about RAG.

This repository gets data about Tokyo Prime Market companies from Wikipedia, so you can ask about things related to that.

## Preparation

1. Create virtual environment

```
python -m venv langchain-env
```

2. Activate virtual environment

```
source langchain-env/bin/activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Create .env

```
OPENAI_API_KEY="your key"
```

5. Get the list of Tokyo Prime Market companies

```
python ./prepare/create_list_from_wiki.py
```

6. Generate embeddings for the data

```
python ./prepare/url_loader.py
```

## Invoke

You can ask questions to modify below in the file "main.py".

```
chain.invoke("極洋株式会社に関連する企業は？")
```

## Sample

Question: 極洋株式会社に関連する企業は？

Answer: "洋株式会社に関連する企業は以下の通りです：\n\n1. 極洋水産株式会社（100%子会社）\n2. 極洋商事株式会社（100%子会社）\n3. 極洋秋津会\n4. 中央魚類株式会社\n\n また、極洋株式会社はあきんどスシローやカッパ・クリエイトとも提携しています。"
