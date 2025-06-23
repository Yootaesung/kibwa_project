import json
import hashlib
import os
import secrets
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, AnyStr
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError, OperationFailure
from bson import ObjectId
from config.settings import settings
from config.logger import logger

class MongoDBManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize MongoDB connection"""
        # Get MongoDB URI from environment variable, or use default external IP
        self.mongo_uri = os.getenv('MONGODB_URI', 'mongodb://3.107.174.223:27017/')  # External MongoDB server connection
        logger.info(f"Connecting to MongoDB at {self.mongo_uri}")
        
        # Add retry logic for connection
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                self.client = MongoClient(
                    self.mongo_uri,
                    serverSelectionTimeoutMS=5000,  # 5 second timeout
                    connectTimeoutMS=10000,         # 10 second connection timeout
                    socketTimeoutMS=45000,          # 45 second socket timeout
                    connect=False,                  # Use lazy connection
                    server_api=ServerApi('1')       # Use Stable API
                )
                
                # Test connection using ping command
                self.client.admin.command('ping')
                logger.info("Successfully connected to MongoDB")
                
                self.db = self.client["member_information"]
                self.users = self.db["users"]
                
                # Create index (username must be unique)
                self.users.create_index("username", unique=True)
                logger.info("Database and indexes are ready")
                return  # Exit method on successful connection
                
            except Exception as e:
                logger.warning(f"MongoDB connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:  # If not the last attempt
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Failed to connect to MongoDB. Application will continue running but database functionality will not be available.")
                    # Application continues running even on connection failure, but database functionality will not be available
                    self.client = None
                    self.db = None
                    self.users = None
    
    def close(self):
        """Close MongoDB connection"""
        if hasattr(self, 'client'):
            self.client.close()
            
# Create global MongoDB manager instance
db_manager = MongoDBManager()

class MemberManager:
    def __init__(self):
        self.db = db_manager
        
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash the password using PBKDF2"""
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        return hashed

    def get_user(self, username: str) -> Optional[Dict]:
        """
        Retrieve user information by username
        
        Args:
            username: Username to query
            
        Returns:
            Optional[Dict]: User information or None
        """
        if not self.db or not self.db.users:
            return None
            
        try:
            user = self.db.users.find_one({"username": username})
            return user
        except Exception as e:
            logger.error(f"Error retrieving user: {str(e)}")
            return None

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Authenticate user credentials.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple[bool, str]: (Success status, Message)
        """
        try:
            user = self.get_user(username)
            if not user:
                return False, "User not found"
                
            salt = user.get("salt")
            if not salt:
                return False, "Invalid user data"
                
            hashed_password = self._hash_password(password, salt)
            if hashed_password == user.get("password"):
                return True, "Authentication successful"
            else:
                return False, "Authentication failed: Incorrect password"
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False, "Authentication error occurred"

    def register(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Register a new member
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple[bool, str]: (Success status, Message)
        """
        try:
            # Duplicate check
            if self.check_member(username):
                return False, "User already exists"

            # Generate password hash and salt
            salt = secrets.token_hex(16)
            hashed_password = self._hash_password(password, salt)

            # Create member data
            member_data = {
                'username': username,
                'password': hashed_password,
                'salt': salt,
                'created_at': datetime.utcnow().isoformat(),
                'last_login': None,
                'is_active': True
            }

            # Save to MongoDB
            try:
                self.db.users.insert_one(member_data)
                logger.info(f"Registration successful: {username}")
                return True, "Registration completed successfully"
            except Exception as e:
                logger.error(f"Error saving to MongoDB: {str(e)}")
                return False, "Error occurred during registration"

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False, f"Registration error: {str(e)}"

            return False, '로그인 처리 중 오류가 발생했습니다.'
            
    def get_user(self, username: str) -> Optional[Dict]:
        """Retrieve user information by username."""
        try:
            user = self.db.users.find_one({'username': username})
            if user:
                # Return only password hash and salt for security reasons
                return {
                    'username': user['username'],
                    'password': user['password'],
                    'salt': user['salt']
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving user information: {str(e)}")
            return None

    def _get_member_key(self, username: str) -> str:
        """Generate key for user document in MongoDB."""
        return f"users/{username}"

    def check_member(self, username: str) -> bool:
        """
        Check if member exists
        
        Args:
            username: Username to check
            
        Returns:
            bool: True if member exists, False otherwise
        """
        try:
            member_key = self._get_member_key(username)
            return self._s3_key_exists(member_key)
        except Exception as e:
            logger.error(f"Error checking member: {str(e)}")
            return False

    def update_session(self, username: str) -> bool:
        """
        Update user session (last login time).
        
        Args:
            username: Username
            
        Returns:
            bool: Update success status
        """
        try:
            # Retrieve user information from MongoDB
            user = self.db.users.find_one({'username': username})
            if not user:
                logger.error(f"Session update failed: User data not found: {username}")
                return False

            # Update last login time
            self.db.users.update_one(
                {'username': username},
                {'$set': {'last_login': datetime.utcnow().isoformat()}}
            )
            logger.info(f"Session update successful: {username}")
            return True
        except Exception as e:
            logger.error(f"Error updating session: {str(e)}")
            return False

# MongoDB를 사용하는 MemberManager 클래스를 기본으로 사용
MemberManager = MemberManager
