from sqlalchemy import Date, Column, ForeignKey, Integer, String, DateTime, Text, PickleType
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    gender = Column(String, index=True)
    bday = Column(Date, index=True)
    password = Column(String)
    created_at = Column(DateTime(), server_default=func.now())
