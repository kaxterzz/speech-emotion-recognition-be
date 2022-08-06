import datetime
from email import message
from itertools import count
from typing import List, Optional

from pydantic import BaseModel




###################################### User ##############################
class UserBase(BaseModel):
    name: str
    email: str
    gender: str
    bday: datetime.date

class User(UserBase):
    user_id: int
    
    class Config:
        orm_mode = True


class UserCreate(UserBase):
    password: str


class UserInDB(UserBase):
    user_id: int
    password: str
    
    class Config:
        orm_mode = True


###################################### Token ##############################
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: int
    name: str
    email: str
    gender: str
    bday: str
    


    
    




    


  

    
      

    



