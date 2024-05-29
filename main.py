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
    text_data = json.loads(body.decode())  # 여기서 json.loads()를 한 번만 호출합니다.
    res = model.classify(text_content=text_data)
    return JSONResponse(res)

@app.post("/api/classify_mt")
async def classify_mt(request: Request):
    body = await request.body()
    text_data = json.loads(body.decode())  # 여기서 json.loads()를 한 번만 호출합니다.
    res = model.classify_mt(text_content=text_data)
    return JSONResponse(res)