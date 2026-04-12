from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Recommender System Prototype")

class RecommendRequest(BaseModel):
    user_id: int
    top_n: int = 10

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[int]

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    # Временный ответ для теста
    return RecommendResponse(user_id=req.user_id, recommendations=[1, 2, 3, 4, 5])

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)