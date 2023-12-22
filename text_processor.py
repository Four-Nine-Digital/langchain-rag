from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

class TextProcessor:
    def __init__(self, encoding_for, encoding, chunk_size, chunk_overlap, separators):
        tiktoken.encoding_for_model(encoding_for)
        self.tokenizer = tiktoken.get_encoding(encoding)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.tiktoken_len,
            separators=separators
        )

    def tiktoken_len(self, text):
        token = self.tokenizer.encode(
            text=text,
            disallowed_special=()
        )

        return len(token)


