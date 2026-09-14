from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader(
    "bio.pdf.pdf"
)
document = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 25
)

chunk = splitter.split_documents(
    document
)

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunk,
    embedding=embeddings,
    collection_name="bio.pdf.pdf"
)
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k" : 3
    }
)

results = retriever.invoke(
    "show my education details"
    "show my skills"
    "show my school details"
)


for rank,doc in enumerate(results, start=1):
    print(f"\nRank Order of Similarlity", {rank})
    print(doc.page_content)
    print("bio data")

