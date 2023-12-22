from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def setupLLM(model_name, context_source):
    llm = Ollama(model=model_name)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=context_source.as_retriever()
    )