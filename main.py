from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.analyze import router

app = FastAPI(title='AI Business Intelligence API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix='/api/v1')

@app.get('/')
def root():
    return {'status': 'running', 'version': '1.0.0'}
