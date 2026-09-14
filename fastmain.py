from fastapi import FastAPI

from model import ChatRequest,ChatResponse
from rag import rag_chain


app = FastAPI()

chat_history=[]

@app.post('/chat',response_model=ChatResponse)
async def Askchatbot(request:ChatRequest):
    # chatbot = Groq(
    #     api_key=os.environ.get("GROQ_API_KEY"),
    # )

    # chat_completion = chatbot.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content":request.chat,
    #         },
    #         {
    #             "role": "system",
    #             "content":request.system_prompt

    #         }
    #     ],
    #     model="openai/gpt-oss-120b",
    # )

    # return (chat_completion.choices[0].message.content)

    answer =rag_chain(
        request.chat,
        request.system_prompt,
        chat_history
    )
    chat_history.append({
        "User: " +request.chat
    })
    chat_history.append({
        "Bot:"+   answer
    })
    return ChatResponse(
        response= answer
    )