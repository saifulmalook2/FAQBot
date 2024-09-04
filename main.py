import asyncio
from helpers import generate_response
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
import logging
from fastapi.encoders import jsonable_encoder 

logging.basicConfig(format="%(levelname)s     %(message)s", level=logging.INFO)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)



app = FastAPI()

@app.get("/")
async def root():
    return {"msg": "OK"}


class faq(BaseModel):
    uid :str
    question: str

@app.post("/ask_faq")
async def faq_response(data: faq):
    data_doc = jsonable_encoder(data)
    question = data_doc['question']
    uid = data_doc['uid']
    response = await generate_response(uid, question)
    return response
    
