import sys
sys.path.append('./')

from fastapi.exceptions import HTTPException
from core.config import get_settings
import pymongo

## Global config variabels
settings = get_settings()


###
### TODO: Batch processiing
###
# async def get_batch(db, pipeline):
#     token_cursors = db['refined'].aggregate(pipeline)
#     async for batch in token_cursors.batch_size(8000):
#         yield batch


def data_generator(collection:str, aggregate:list, batch_size:int=10000):
    with pymongo.MongoClient(settings.DATABASE_URL) as client:
        db = client['imcp']
        documents = db[collection].aggregate(aggregate).batch_size(batch_size)
        batch = []
        for doc in documents:
            batch.append(doc)
            if len(batch) == batch_size:
                yield batch  # Trả về nhóm tài liệu (batch)
                batch = []  # Reset batch sau khi yield
        # Nếu còn tài liệu dư ra sau khi lặp xong
        if batch:
            yield batch


###
### TODO: Get "text + token" for caption and short_caption
###
async def getCaptionTokens(page:int=1, per_page:int=8000):
    tokens = []
    try:
        pipeline = [{
                '$sort': {'url': 1}
            }, {
                '$project': {'created_time': 0, 'publisher': 0, '_id': 0 }
            }, {
                '$skip': (page-1)*per_page
            }, {
                '$limit': per_page
            }
        ]
        for batch in data_generator('refined', pipeline, per_page):
            tokens += batch
    except HTTPException:
        raise HTTPException(status_code=505, detail="Fetching is failed - watch in internal")
    return tokens


###
### TODO: Get only tokens for caption and short_caption
###
async def getOnlyTokens(page:int=1, per_page:int=8000):
    tokens = []
    try:
        pipeline = [{
                '$sort': {'url': 1}
            }, {
                '$project': {'_id':0, 'caption_tokens':1, 'short_caption_tokens':1, 'url':1}
            }, {
                '$skip': (page-1)*per_page
            }, {
                '$limit': per_page
            }
        ]
        for batch in data_generator('refined', pipeline, per_page):
            tokens += batch
    except HTTPException:
        raise HTTPException(status_code=505, detail="Fetching is failed - watch in internal")
    return tokens


###
### TODO: Get only text for caption and short_caption
###
async def getOnlyTexts(page:int=1, per_page:int=8000):
    tokens = []
    try:
        pipeline = [{
                '$sort': {'url': 1}
            }, {
                '$project': {'_id':0, 'caption':1, 'short_caption':1, 'url':1}
            }, {
                '$skip': (page-1)*per_page
            }, {
                '$limit': per_page
            }
        ]
        for batch in data_generator('refined', pipeline, per_page):
            tokens += batch
    except HTTPException:
        raise HTTPException(status_code=505, detail="Fetching is failed - watch in internal")
    return tokens


###
### TODO: Get "text + token" for caption and short_caption of android
###
async def getCaptionTokensMobile(page:int=1, per_page:int=8000):
    tokens = []
    try:
        pipeline = [{
                '$match': {'publisher': 'android'}
            }, {
                '$sort': {'created_time': -1}
            }, {
                '$project': {'created_time': 0, 'publisher': 0, '_id': 0 }
            }, {
                '$skip': (page-1)*per_page
            }, {
                '$limit': per_page
            }
        ]
        for batch in data_generator('refined', pipeline, per_page):
            tokens += batch
    except HTTPException:
        raise HTTPException(status_code=505, detail="Fetching is failed - watch in internal")
    return tokens