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
        """MongoDB 연결 초기화"""
        # 환경 변수에서 MongoDB URI를 가져오고, 없으면 기본값으로 외부 IP 사용
        self.mongo_uri = os.getenv('MONGODB_URI', 'mongodb://3.107.174.223:27017/')  # 외부 MongoDB 서버 연결
        logger.info(f"Connecting to MongoDB at {self.mongo_uri}")
        
        # 연결 재시도 로직 추가
        max_retries = 3
        retry_delay = 2  # 초 단위
        
        for attempt in range(max_retries):
            try:
                self.client = MongoClient(
                    self.mongo_uri,
                    serverSelectionTimeoutMS=5000,  # 5초 타임아웃
                    connectTimeoutMS=10000,         # 연결 타임아웃 10초
                    socketTimeoutMS=45000,          # 소켓 타임아웃 45초
                    connect=False,                  # 지연 연결 사용
                    server_api=ServerApi('1')       # Stable API 사용
                )
                
                # 연결 테스트 (ping 명령 사용)
                self.client.admin.command('ping')
                logger.info("Successfully connected to MongoDB")
                
                self.db = self.client["member_information"]
                self.users = self.db["users"]
                
                # 인덱스 생성 (username은 고유해야 함)
                self.users.create_index("username", unique=True)
                logger.info("Database and indexes are ready")
                return  # 연결 성공 시 메서드 종료
                
            except Exception as e:
                logger.warning(f"MongoDB 연결 시도 {attempt + 1}/{max_retries} 실패: {e}")
                if attempt < max_retries - 1:  # 마지막 시도가 아니라면
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 지수 백오프
                else:
                    logger.error("MongoDB에 연결할 수 없습니다. 애플리케이션을 계속 실행하지만 데이터베이스 기능은 사용할 수 없습니다.")
                    # 연결 실패 시에도 애플리케이션은 계속 실행되지만, 데이터베이스 기능은 사용할 수 없음
                    self.client = None
                    self.db = None
                    self.users = None
    
    def close(self):
        """MongoDB 연결 종료"""
        if hasattr(self, 'client'):
            self.client.close()
            
# 전역 MongoDB 매니저 인스턴스 생성
db_manager = MongoDBManager()

class MemberManager:
    def __init__(self):
        self.db = db_manager
        
    def _hash_password(self, password: str, salt: str) -> str:
        """비밀번호를 해시화합니다."""
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        return hashed

    def get_user(self, username: str) -> Optional[Dict]:
        """
        사용자명으로 사용자 정보를 조회합니다.
        
        Args:
            username: 조회할 사용자명
            
        Returns:
            Optional[Dict]: 사용자 정보 또는 None
        """
        if not self.db or not self.db.users:
            return None
            
        try:
            user = self.db.users.find_one({"username": username})
            return user
        except Exception as e:
            logger.error(f"사용자 조회 중 오류 발생: {str(e)}")
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """
        사용자 인증을 수행합니다.
        
        Args:
            username: 사용자명
            password: 비밀번호
            
        Returns:
            Optional[Dict]: 인증된 사용자 정보 또는 None
        """
        if not self.db or not self.db.users:
            logger.error("데이터베이스 연결이 설정되지 않았습니다.")
            return None
            
        try:
            # 사용자 조회
            user = self.get_user(username)
            if not user:
                logger.warning(f"사용자를 찾을 수 없음: {username}")
                return None
            
            # 비밀번호 검증
            salt = user.get("salt")
            hashed_password = self._hash_password(password, salt)
            
            if hashed_password == user.get("password"):
                return user
            else:
                logger.warning(f"비밀번호 불일치: {username}")
                return None
                
        except Exception as e:
            logger.error(f"인증 중 오류 발생: {str(e)}")
            return None

    def register(self, username: str, password: str) -> Tuple[bool, str]:
        """
        새로운 회원을 등록합니다.
        
        Args:
            username: 사용자명
            password: 비밀번호
            
        Returns:
            Tuple[bool, str]: (성공 여부, 메시지)"""
        """
        try:
            # 중복 체크
            if self.check_member(username):
                return False, "이미 등록된 사용자입니다."

            # 비밀번호 해시화와 salt 생성
            salt = secrets.token_hex(16)
            hashed_password = self._hash_password(password, salt)

            # 사용자 데이터 생성
            member_data = {
                'username': username,
                'password': hashed_password,
                'salt': salt,
                'created_at': datetime.utcnow().isoformat(),
                'last_login': None,
                'is_active': True
            }

            # MongoDB에 저장
            try:
                self.db.users.insert_one(member_data)
                logger.info(f"회원가입 성공: {username}")
                return True, "회원가입이 완료되었습니다."
            except Exception as e:
                logger.error(f"MongoDB 저장 중 오류 발생: {str(e)}")
                return False, "회원가입 중 오류가 발생했습니다."

        except Exception as e:
            logger.error(f"회원가입 중 오류 발생: {e}")
            return False, f"회원가입 중 오류가 발생했습니다: {str(e)}"

    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """
        회원 로그인을 처리합니다.
        
        Args:
            username: 사용자명
            password: 비밀번호
            
        Returns:
            Tuple[bool, str]: (성공 여부, 메시지)
        """
        try:
            # 사용자 조회
            user = self.get_user(username)
            if not user:
                return False, "사용자가 존재하지 않습니다."

            # 비밀번호 검증
            salt = user.get('salt')
            if not salt:
                return False, "비밀번호 검증에 실패했습니다."

            hashed_password = self._hash_password(password, salt)
            if user['password'] != hashed_password:
                return False, "비밀번호가 일치하지 않습니다."

            # 로그인 성공
            self.update_session(username)  # 세션 업데이트
            logger.info(f"로그인 성공: {username}")
            return True, "로그인 성공"
                
        except Exception as e:
            logger.error(f"로그인 중 오류 발생: {str(e)}")
            return False, '로그인 처리 중 오류가 발생했습니다.'
            
    def get_user(self, username: str) -> Optional[Dict]:
        """사용자명으로 사용자 정보를 조회합니다."""
        try:
            user = self.db.users.find_one({'username': username})
            if user:
                # 비밀번호 해시와 salt만 반환 (보안상의 이유)
                return {
                    'username': user['username'],
                    'password': user['password'],
                    'salt': user['salt']
                }
            return None
        except Exception as e:
            logger.error(f"사용자 정보 조회 중 오류: {str(e)}")
            return None

    def _get_member_key(self, username: str) -> str:
        """MongoDB에서 사용자 문서의 키를 생성합니다."""
        return f"users/{username}"

    def check_member(self, username: str) -> bool:
        """
        회원 존재 여부를 확인합니다.
        
        Args:
            username: 확인할 사용자명
            
        Returns:
            bool: 회원이 존재하면 True, 그렇지 않으면 False
        """
        try:
            member_key = self._get_member_key(username)
            return self._s3_key_exists(member_key)
        except Exception as e:
            logger.error(f"회원 확인 중 오류 발생: {str(e)}")
            return False

    def update_session(self, username: str) -> bool:
        """
        사용자 세션을 업데이트합니다 (마지막 로그인 시간 갱신).
        
        Args:
            username: 사용자명
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            # MongoDB에서 사용자 정보 조회
            user = self.db.users.find_one({'username': username})
            if not user:
                logger.error(f"세션 업데이트 실패: 사용자 데이터를 찾을 수 없습니다: {username}")
                return False

            # 마지막 로그인 시간 업데이트
            self.db.users.update_one(
                {'username': username},
                {'$set': {'last_login': datetime.utcnow().isoformat()}}
            )
            logger.info(f"세션 업데이트 성공: {username}")
            return True
        except Exception as e:
            logger.error(f"세션 업데이트 중 오류 발생: {str(e)}")
            return False

# MongoDB를 사용하는 MemberManager 클래스를 기본으로 사용
MemberManager = MemberManager
