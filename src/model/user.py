from sqlalchemy import Column, Integer, String, ForeignKey, Enum, create_engine
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import TEXT
import enum

Base = declarative_base()

# Role Enum remains unchanged
class RoleEnum(enum.Enum):
    admin = "admin"
    user = "user"

# Updated User Model
class User(Base):
    __tablename__ = "users"

    id = Column(String(225), primary_key=True)
    display_name = Column(String(225), nullable=False)
    email = Column(String(225), unique=True, nullable=False)
    photo_url = Column(String(225))
    access_token = Column(String(225))
    id_token = Column(String(225))
    role = Column(Enum(RoleEnum), nullable=False, default=RoleEnum.user)
    history = relationship("UserHistory", back_populates="user")

    def __repr__(self):
        return (
            f"User(id={self.id}, display_name={self.display_name}, "
            f"email={self.email}, photo_url={self.photo_url}, role={self.role})"
        )

class Playlist(Base):
    __tablename__ = "playlists"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)  # Playlist name
    age_group = Column(String(50), nullable=False)  # Example: "Kids", "Teens", "Adults"
    spotify_id = Column(String(50), nullable=False)  # Spotify playlist ID
    description = Column(TEXT, nullable=True)  # Optional description
    genre = Column(String(50), nullable=True)  # Genre classification
    popularity = Column(Integer, nullable=True, default=0)  # A score for ranking


# User History Table
class UserHistory(Base):
    __tablename__ = "user_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(225), ForeignKey("users.id"), nullable=False)
    playlist_id = Column(Integer, ForeignKey("playlists.id"), nullable=False)
    timestamp = Column(String(50), nullable=False)  # Timestamp for activity tracking
    user = relationship("User", back_populates="history")
    playlist = relationship("Playlist", back_populates="history")
    
class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(225), ForeignKey("users.id"), nullable=False)
    playlist_id = Column(Integer, ForeignKey("playlists.id"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1 to 5 stars

class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)

class PlaylistTag(Base):
    __tablename__ = "playlist_tags"
    playlist_id = Column(Integer, ForeignKey("playlists.id"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), primary_key=True)

class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(225), ForeignKey("users.id"), nullable=False)
    preferred_genre = Column(String(50), nullable=True)
    preferred_age_group = Column(String(50), nullable=True)

# Database Setup
DATABASE_URL = "mysql+pymysql://username:password@localhost/spotify_app"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables
Base.metadata.create_all(bind=engine)
