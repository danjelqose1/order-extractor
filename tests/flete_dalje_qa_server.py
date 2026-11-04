import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import db
from flete_dalje import router
app=FastAPI()
db.init_db()
app.include_router(router)
app.mount('/',StaticFiles(directory=str(ROOT / 'docs'),html=True))
