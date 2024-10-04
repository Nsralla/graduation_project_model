from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {'message':'Hello, world'};

@app.get("/hello/{name}")
def greeting(name:str):
    return {"message":f"hello, {name}"}

@app.get("/items")
def get_items():
    return {"message":"items"}
@app.get("/items/{item_id}")
def get_item(item_id:int):
    return {"message":item_id}
