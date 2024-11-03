import sys
sys.path.append('./')

from core.security import oauth2_scheme
from fastapi import status, APIRouter, Depends
from fastapi.responses import JSONResponse
from src.users.models import User
from src.users.services import createUser, getUser


# router
user_router = APIRouter(
    prefix="/api/v1/users",
    tags=["Users"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(oauth2_scheme)]
)


########## Endpoints ##########

@user_router.post("/create-user", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: User):
    user = await createUser(user)
    payload = {
        "message": "User account has been successfully created.",
        "data": user
    }
    return JSONResponse(content=payload)


@user_router.get("/get-user/{username}", response_model=User, status_code=status.HTTP_200_OK)
async def get_user(username: str):
    user = await getUser(username)
    payload = {
        "message": "User account has been existing.",
        "data": user
    }
    return JSONResponse(content=payload)
     