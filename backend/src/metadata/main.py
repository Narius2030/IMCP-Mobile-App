from fastapi import FastAPI
from fastapi.responses import JSONResponse
from core.auth.routes import auth_router
from routes import metadata_router
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.middleware.cors import CORSMiddleware
from core.security import JWTAuth

app = FastAPI(
    title="Metadata API", 
    docs_url='/docs', 
    openapi_url="/openapi.json"
)
app.include_router(auth_router)
app.include_router(metadata_router)

# Add Middleware
origins = ['*']
app.add_middleware(AuthenticationMiddleware, backend=JWTAuth())
app.add_middleware( 
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get('/')
def health_check():
    return JSONResponse(content={'status':'running'})