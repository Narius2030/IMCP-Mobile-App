import sys
sys.path.append('./')

from src.users.models import User
from fastapi.exceptions import HTTPException
from core.security import get_password_hash
from core.config import get_settings
import pymongo


settings = get_settings()

async def createUser(user:User):
    with pymongo.MongoClient(settings.DATABASE_URL) as client:
        db = client['imcp']
        # Hash password
        user.password = get_password_hash(user.password)
        # Create a new user account
        collection = db['user_account']
        try:
            result = collection.insert_one(user.model_dump())
            inserted_user = collection.find_one({"_id": result.inserted_id}, {"password":0, "_id":0})
        except Exception as exc:
            raise Exception(str(exc))

    return inserted_user


async def getUser(username:str):
    with pymongo.MongoClient(settings.DATABASE_URL) as client:
        db = client['imcp']
        collection = db['user_account']
        # Find the account with entered full_name
        user = collection.find_one({"username": username}, {"password":0, "_id":0})
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
    
    return user