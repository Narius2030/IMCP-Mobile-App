from fastapi import FastAPI
from fastapi.responses import JSONResponse
from routes import caption_router
# from starlette.middleware.authentication import AuthenticationMiddleware
# from starlette.middleware.cors import CORSMiddleware
# from core.security import JWTAuth

app = FastAPI(
    title="Metadata API", 
    docs_url='/api/v1/metadata/docs', 
    openapi_url="/api/v1/metadata/openapi.json")
# app.include_router(auth_router)
app.include_router(caption_router)

# Add Middleware
origins = ['*']

@app.get('/')
def health_check():
    return JSONResponse(content={'status':'running'})