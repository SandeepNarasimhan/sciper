from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root():
    return 'Default get root \n Hello API'

class inputfeatures(BaseModel):
    name: str
    score: float
    id: int

@app.post("/predict")
def predict(data: inputfeatures, request: Request):
    print(request.url)
    df = data.dict()
    output = {'id':df['id'],
    'name': df['name'],
    'score': df['score']}
    return f"{df['name']} scored {df['score']} and the id is {df['id']+10}"
