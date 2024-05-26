from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from service.model import NLPModel
import json

app = FastAPI()
model = NLPModel()

@app.get("/")
async def root():
    return {"message":"Hello World"}

@app.api_route("/ping", methods=["GET", "POST", "PUT", "PATCH","DELETE", "HEAD", "OPTIONS"])
async def ping():
    return Response(status_code=200)

@app.post("/api/classify")
async def classify(request: Request):
    body = await request.body()
    text_data = json.loads(json.loads(body.decode()))
    # json array -> python list
    res = model.classify(text_content = text_data)
    # batch-size를 4개로 할당해 CPU 환경에서의 성능 향상을 고려했습니다.
    return JSONResponse(res)