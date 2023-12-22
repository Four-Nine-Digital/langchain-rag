# Retrieval (RAG) and QA Answering Architecture

This is a simple architecture designed for retrieval and QA answering. The system leverages the following components:

- **Pinecone:** A service for storing and querying high-dimensional vectors efficiently.

- **Ollama:** A service that allows you to run open-source large language models, such as Llama 2, locally. To utilize Ollama as the language model (LLM) for running this application, ensure you download and install Ollama. Visit the following link: [ollama.ai](https://ollama.ai/)]. You also need to pull llama2 model.

- **LangChain:** A language processing library for various natural language processing tasks.

- **Embedding Engines:** You have the flexibility to choose between two powerful embedding engines for text representation:

  - **OpenAI Embedding:** Utilizes OpenAI's embedding model for text representation. Note: To use this with OpenAI, you must have an OpenAI API key. OpenAI is a paid service.

  - **Hugging Face Embedding:** Utilizes Hugging Face's embedding model for text representation.
