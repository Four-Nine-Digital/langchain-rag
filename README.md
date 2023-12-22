# Retrieval (RAG) and QA Answering Architecture

This is a simple architecture designed for retrieval and QA answering. The system leverages the following components:
- **Dataset:** The dataset used in this example is sourced from the Hugging Face datasets. If you wish to experiment with an alternative dataset, you can explore the Hugging Face datasets page and modify the load_dataset arguments to incorporate the dataset of your preference.
- **Pinecone:** A service for storing and querying high-dimensional vectors efficiently.

- **Ollama:** A service that allows you to run open-source large language models, such as Llama 2, locally. To utilize Ollama as the language model (LLM) for running this application, ensure you download and install Ollama. Visit the following link: [ollama.ai](https://ollama.ai/). You also need to pull llama2 model.

- **LangChain:** A language processing library for various natural language processing tasks.

- **Embeddings:** You have the flexibility to choose between two powerful embedding engines for text representation:

  - **OpenAI Embeddings:** Utilizes OpenAI's embedding model for text representation. Note: To consume OpenAI APIs, you must have an OpenAI API key (OpenAI is a paid service).

  - **Hugging Face Embeddings:** Utilizes Hugging Face's embedding model for text representation.
