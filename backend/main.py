from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.users.routes import user_router
from src.auth.routes import auth_router
from src.captions.routes import caption_router
from src.generation.routes import generation_router
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.middleware.cors import CORSMiddleware
from core.security import JWTAuth

app = FastAPI(docs_url='/docs')
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(caption_router)
app.include_router(generation_router)

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