from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from service.model import NLPModel

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
    text_data = body.decode()
    text_data = text_data.split(",")
    # 복수개의 문장을 동시 처리할 수 있도록 구분자를 추가했습니다.
    res = model.classify(text_content = text_data, batch_size=4)
    #batch-size를 4개로 할당해 CPU 환경에서의 성능 향상을 고려했습니다.
    return JSONResponse(res)