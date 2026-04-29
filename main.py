from fastapi import FastAPI
from routers.analyze import router

app = FastAPI(title='AI Business Intelligence API')

app.include_router(router, prefix='/api/v1')


@app.get('/')
def root():
    return {'status': 'running', 'version': '1.0.0'}
