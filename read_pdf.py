from docling.document_converter import DocumentConverter
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MINIO_USERNAME = os.getenv("MINIO_USERNAME")
MINIO_PASSWORD = os.getenv("MINIO_PASSWORD")

# Convert PDFs
pdf_dir = "data"
converter = DocumentConverter()
all_docs = []

for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        print(f"Converting {file}")
        result = converter.convert(os.path.join(pdf_dir, file))
        doc = result.document
        markdown_text = doc.export_to_markdown()
        all_docs.append(Document(page_content=markdown_text, metadata={"source": file}))

# Chunk text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n##", "\n#", "\n- ", "\n", " "]
)
chunks = splitter.split_documents(all_docs)
print(f"Total chunks: {len(chunks)}")

# Store in Milvus
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY
)

vectorstore = Milvus.from_documents(
    chunks,
    embeddings,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name="pdf_rag_docs"
)

print("PDF chunks stored in Milvus")

# Query with retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

query = "List all the acronyms in the document"

results = retriever.get_relevant_documents(query)

#for i, doc in enumerate(results):
    #print(f"\nResult {i+1} (from {doc.metadata['source']}):\n{doc.page_content[:500]}")

# OpenAI GPT-4o-mini for QA
print("Initializing OpenAI GPT-4o-mini")
try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.1
    )
except Exception as e:
    print(f"Error initializing OpenAI LLM: {e}")
    raise

try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    query = "List all the acronyms in the document"
    result = qa_chain.invoke(query)

    print("Answer:", result["result"])
    print("Sources:", [doc.metadata["source"] for doc in result["source_documents"]])
    
except Exception as e:
    print(f"Error during query processing: {e}")
    raise
