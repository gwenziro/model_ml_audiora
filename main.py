import os
import logging
from datetime import datetime
from fastapi import HTTPException, Depends, File, UploadFile, FastAPI
from sqlalchemy.orm import Session
from src.model.database import get_db
from src.model.user import User, Playlist, UserHistory, RoleEnum
from ML.lbp_utils import predict_age_group
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


app = FastAPI()

@app.post("/recommend_playlist")
async def recommend_playlist_route(user_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Ensure upload directory exists
    upload_dir = "ML/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    safe_filename = os.path.basename(file.filename)  # Sanitize filename
    file_path = os.path.join(upload_dir, safe_filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Call recommendation logic
    try:
        result = recommend_playlist(user_id=user_id, db=db, image_path=file_path)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def recommend_playlist(user_id: str, db: Session = Depends(get_db), image_path: str):
    # Fetch user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.role == RoleEnum.admin:
        return {"message": "Admins do not receive recommendations"}

    # Predict age group using LBP
    try:
        predicted_age_group = predict_age_group(image_path)
    except Exception as e:
        logging.error(f"Age prediction failed for {image_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Age prediction failed: {str(e)}")

    # Find playlist for the predicted age group
    playlist = db.query(Playlist).filter(Playlist.age_group == predicted_age_group).first()
    if not playlist:
        raise HTTPException(status_code=404, detail=f"No playlist found for age group '{predicted_age_group}'.")

    # Log user history
    history = UserHistory(
        user_id=user.id,
        playlist_id=playlist.id,
        timestamp=datetime.now().isoformat()
    )
    db.add(history)
    db.commit()

    return {"playlist_id": playlist.spotify_id, "description": playlist.description}
