from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0.2
)


loader = PyPDFLoader(
    "Week2_Tasks.pdf"
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
    collection_name="Week2_Tasks.pdf"
)
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k" : 3
    }
)

def format_docs(docs):
    return "\n\n".join(
    doc.page_content
    for doc in docs
)
chat_history = []

rag_prompt = ChatPromptTemplate.from_template("""
You are a Personal bio assistant.

Answer the question using only the information
provided in the context.
Conversation History:
{chat_history}

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say "I don't know."
""")


def format_history(history):
    return "\n".join(
        f"User: {item['user']}\nAssistant: {item['assistant']}"
        for item in history
    )

rag_chain = (
        {
        "context": RunnableLambda(
            lambda x: retriever.invoke(x["question"])
        ) | format_docs,

        "question": RunnableLambda(
            lambda x: x["question"]
        ),

        "chat_history": RunnableLambda(
            lambda x: format_history(x["chat_history"])
        )
    }
    | rag_prompt
    | llm
)
while True:

        question = input("Ask Week2_Tasks: ")


        if question.lower()=="complete":
            break


        answer = rag_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

        print(answer.content)

        chat_history.append(
        {
            "user" : question,
            "assistant" : answer.content
        }
    )

        print(chat_history)