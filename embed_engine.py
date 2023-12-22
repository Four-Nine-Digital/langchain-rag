from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from constants import HUGGING_FACE, OPENAI_EMBEDDING

def embed_engine_factory(engine_type):
    if engine_type == OPENAI_EMBEDDING:
        return OpenAIEmbeddings(
            model='MODEL_NAME',
            openai_api_key='OPENAI_API_KEY'
        )
    elif engine_type == HUGGING_FACE:
        return HuggingFaceEmbeddings()
    else:
        # Raise an error for unsupported engine_type
        raise ValueError(f'Error: Engine type "{engine_type}" is not compatible.')

        