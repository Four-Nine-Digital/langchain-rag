from langchain.vectorstores import Pinecone
from constants import OPENAI_EMBEDDING
from text_processor import TextProcessor
from embed_engine import embed_engine_factory
from uuid import uuid4
import pinecone

class PineconeClient:
    def __init__(self, api_key, api_env, index_name, metric, embed_engine):
        self.__openai_dimension = 1536 # dimensions text-embedding-ada-002 produces vectors with 1536 dimensions
        self.__hf_dimension = 768 # dimensions HF produces vectors with 1536 dimensions

        # Initialize pinecone client
        pinecone.init(
        api_key=api_key,
        environment=api_env
        )

        # Create index 
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
            name=index_name,
            metric=metric,
            dimension=self.__openai_dimension if embed_engine == OPENAI_EMBEDDING else self.__hf_dimension
            )
        
        # Setup pinecone index
        self.index = pinecone.GRPCIndex(index_name)
        self.vectorstore_index = pinecone.Index(index_name)

        # Setup text processor and embeddings engine
        self.text_processor = TextProcessor(
        encoding_for='gpt-3.5-turbo',
        encoding='cl100k_base',
        chunk_size=400,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]
        )
        self.embed = embed_engine_factory(embed_engine)

        # Setup vectorstore
        self.vectorstore = Pinecone(
            self.vectorstore_index,
            self.embed.embed_query,
            'text'
        )

    def get_index_stats(self):
        return self.index.describe_index_stats()
    
    def is_database_populated(self):
        stats = self.get_index_stats()
        return stats['total_vector_count'] > 0
    
    def create_index(self, dataset, batch_limit=100):
        texts = []
        metadatas = []
        for _, record in enumerate(dataset):
            # first get metadata fields for this record
            metadata = {
                'id': str(record['id']),
                'source': record['source']
            }
            # now we create chunks from the record text
            record_texts = self.text_processor.text_splitter.split_text(record['text'])
            # create individual metadata dicts for each chunk
            record_metadatas = [{
                "chunk": j, "text": text, **metadata
            } for j, text in enumerate(record_texts)]
            # append these to current batches
            texts.extend(record_texts)
            metadatas.extend(record_metadatas)
            # if we have reached the batch_limit we can add texts
            if len(texts) >= batch_limit:
                ids = [str(uuid4()) for _ in range(len(texts))]
                embeds = self.embed.embed_documents(texts)
                self.index.upsert(vectors=zip(ids, embeds, metadatas))
                texts = []
                metadatas = []

        if len(texts) > 0:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = self.embed.embed_documents(texts)
            self.index.upsert(vectors=zip(ids, embeds, metadatas)) 