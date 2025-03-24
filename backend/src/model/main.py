from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.model.routes import model_router
# from starlette.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Model API", 
    docs_url='/api/v1/model/docs', 
    openapi_url="/api/v1/model/openapi.json"
)
app.include_router(model_router)

# Add Middleware
origins = ['*']

@app.get('/')
def health_check():
    return JSONResponse(content={'status':'running'})