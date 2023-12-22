from pinecone_client import PineconeClient
from llm import setupLLM
from datasets import load_dataset
from dotenv import load_dotenv
from constants import HUGGING_FACE
import os

load_dotenv()

# Pinecone
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
embeddings_model_metric = "cosine" # you can use "euclidean", "manhattan", "hamming", etc.

# Load dataset from HF
dataset = load_dataset("gavmac00/nextjs-app-docs", split="train")

pci = PineconeClient(
    api_key=PINECONE_API_KEY,
    api_env=PINECONE_ENVIRONMENT,
    index_name='nextjs-docs-retrieval-augmentation',
    metric=embeddings_model_metric,
    embed_engine=HUGGING_FACE
)

# indexing
if not pci.is_database_populated():
    pci.create_index(dataset)

qa = setupLLM(model_name="llama2", context_source=pci.vectorstore)

while True:
    query = input('Ask your Next.js Question here (type "exit" to quit): ')

    if query.lower() == 'exit':
        print('Exiting the program. Goodbye!')
        break

    print(qa.run(query))

