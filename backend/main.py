from fastapi import FastAPI

app = FastAPI(title="FilingSentry Agent API")

@app.get("/")
def health_check():
    return {
        "message": "FilingSentry Agent backend is running"
    }