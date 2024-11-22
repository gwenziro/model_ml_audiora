from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from ..main import recommend_playlist, search_playlists
from ..src.model.database import get_db

app = FastAPI()

@app.post("/recommend_playlist")
def recommend_playlist_route(user_id: str, db: Session = Depends(get_db)):
    return recommend_playlist(user_id, db)

@app.get("/search_playlists")
def search_playlists_route(query: str = None, age_group: str = None, genre: str = None, db: Session = Depends(get_db)):
    return search_playlists(query, age_group, genre, db)
