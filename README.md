# Question&Answering and RAG with TextToSQL on TabularData
## Experimenting with tabular data using TextToSQL and RAG techniques

 ![Badge en Desarollo](https://img.shields.io/badge/STATUS-EN%20DESAROLLO-green)

## This repository is still in progress

## Description

This project is a Gen AI project that utilizes Open AI GPT 4, Langchain, SQLite, and Milvus and allows users to interact (perform Q&A and RAG) with SQL databases, or CSV files using natural language.

Techniques in progress:
- Chat with tabular data in a SQL database (SQLite)
- Chat with preprocessed CSV data ingested into a SQL database
- Chat wuth Tabular data using Text-To-SQL LLM agent
- RAG with tabular data

## Models

In this repo, we are experimenting with OpenAI Langchain Agents using the OpenAI model "gpt-4"

## Generative AI Topics

- LLM chains and agents
- GPT function calling
- Retrieval Augmented generation (RAG)

## Content

You can find the source code in the [src](./src/) folder:
- AIModels: class to instantiate the LLM model and the embeddings.
- SQLAgent and TextToSQLAgent: agents to query tabular data in a SQL database using Langchain SQL agents and chains.
- RAGTabularDataAgent: an agent to query tabular data using a RAG approach.
- VectorsFromTabularData: class to ingest tabular data in CSV format into a Milvus database, then a RAG agent will query this data.
- SQLDBFromTabularData: build a SQL database from CSV files.

## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

Copyright 2023 Eduardo Muñoz

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
