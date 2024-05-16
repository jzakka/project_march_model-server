from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from service.model import NLPModel
app = FastAPI()

model = NLPModel()

@app.get("/")
async def root():
    return {"message":"Hello World"}

@app.post("/api/classify")
async def classify(request: Request):
    body = await request.body()
    text_data = body.decode()
    return JSONResponse(model.classify(text_content = text_data))