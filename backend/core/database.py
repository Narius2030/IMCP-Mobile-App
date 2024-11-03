import sys
sys.path.append('./')

import pymongo
from core.config import get_settings

settings = get_settings()

class Database():
    def connect(self):
        client = pymongo.MongoClient(settings.DATABASE_URL)
        db = client['imcp']
        return db