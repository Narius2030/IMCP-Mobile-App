import sys
sys.path.append('./')

from fastapi.exceptions import HTTPException
from core.config import get_settings
from motor import motor_asyncio

## Global config variabels
settings = get_settings()


###
### TODO: Batch processiing
###
async def get_batch(db, pipeline):
    token_cursors = db['refined'].aggregate(pipeline)
    async for batch in token_cursors.batch_size(5000):
        yield batch


###
### TODO: Get "text + token" for caption and short_caption
###
async def getCaptionTokens(num_rows):
    tokens = []
    try:
        client = motor_asyncio.AsyncIOMotorClient(settings.DATABASE_URL)
        db = client['imcp']
        pipeline = [{
                '$sort': {'url': 1}
            }, {
                '$project': {'created_time': 0, 'publisher': 0, '_id': 0 }
            }, {
                '$limit': int(num_rows)
            }
        ]
        async for batch in get_batch(db, pipeline):
            tokens.append(batch)
    except HTTPException:
        raise HTTPException(status_code=505, detail="Fetching is failed - watch in internal")
    return tokens


###
### TODO: Get only tokens for caption and short_caption
###
async def getOnlyTokens(num_rows):
    tokens = []
    try:
        client = motor_asyncio.AsyncIOMotorClient(settings.DATABASE_URL)
        db = client['imcp']
        pipeline = [{
                '$sort': {'url': 1}
            }, {
                '$project': {'_id':0, 'caption_tokens':1, 'short_caption_tokens':1, 'url':1}
            }, {
                '$limit': int(num_rows)
            }
        ]
        async for batch in get_batch(db, pipeline):
            tokens.append(batch)
    except HTTPException:
        raise HTTPException(status_code=505, detail="Fetching is failed - watch in internal")
    return tokens


###
### TODO: Get only text for caption and short_caption
###
async def getOnlyTexts(num_rows):
    tokens = []
    try:
        client = motor_asyncio.AsyncIOMotorClient(settings.DATABASE_URL)
        db = client['imcp']
        pipeline = [{
                '$sort': {'url': 1}
            }, {
                '$project': {'_id':0, 'caption':1, 'short_caption':1, 'url':1}
            }, {
                '$limit': int(num_rows)
            }
        ]
        async for batch in get_batch(db, pipeline):
            tokens.append(batch)
    except HTTPException:
        raise HTTPException(status_code=505, detail="Fetching is failed - watch in internal")
    return tokens