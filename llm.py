from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def setupLLM(retriever):
    llm = Ollama(model="llama2")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever.as_retriever()
    )