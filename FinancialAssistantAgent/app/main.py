from fastapi import FastAPI
from app.models import ChatRequest, ChatResponse
from app.logger import log_interaction
from finance_chat_Term import chat_fn, DEFAULT_SYSTEM_PROMPT  # ✅ Already exists in your code

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # ✅ Log incoming request
    log_interaction(user_id=request.user_id, user_message=request.message, ai_response=None)

    # ✅ Call your actual LLM logic
    llm_response = chat_fn(
        message=request.message,
        history=[],  # later we plug memory here
        system_prompt=DEFAULT_SYSTEM_PROMPT
    )

    # ✅ Log AI reply
    log_interaction(user_id=request.user_id, user_message=request.message, ai_response=llm_response)

    return ChatResponse(response=llm_response)
