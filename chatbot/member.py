import json
import hashlib
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from pymongo import MongoClient
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
        self.mongo_uri = "mongodb://3.107.174.223:27017/"
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client["member_information"]
        self.users = self.db["users"]
        
        # 인덱스 생성 (username은 고유해야 함)
        self.users.create_index("username", unique=True)
    
    def close(self):
        """MongoDB 연결 종료"""
        if hasattr(self, 'client'):
            self.client.close()
            
# 전역 MongoDB 매니저 인스턴스 생성
db_manager = MongoDBManager()

class MemberManager:
    def __init__(self):
        self.db = db_manager
        
    def _hash_password(self, password: str) -> Tuple[str, str]:
        """비밀번호를 해시화합니다."""
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        return hashed, salt

    def register(self, username: str, password: str) -> Tuple[bool, str]:
        """
        새로운 회원을 등록합니다.
        
        Args:
            username: 사용자명
            password: 비밀번호
            
        Returns:
            Tuple[bool, str]: (성공 여부, 메시지)
        """
        try:
            # 비밀번호 해시화
            hashed_password, salt = self._hash_password(password)
            
            # 회원 정보 생성
            user_data = {
                'username': username,
                'password': hashed_password,
                'salt': salt,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            
            # MongoDB에 저장
            self.db.users.insert_one(user_data)
            
            return True, "회원가입이 완료되었습니다."
            
        except DuplicateKeyError:
            return False, "이미 존재하는 사용자명입니다."
        except Exception as e:
            logger.error(f"회원가입 중 오류 발생: {e}")
            return False, f"회원가입 중 오류가 발생했습니다: {str(e)}"

    def login(self, username: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        회원 로그인을 처리합니다.
        
        Args:
            username: 사용자명
            password: 비밀번호
            
        Returns:
            Tuple[bool, str, Optional[Dict]]: (성공 여부, 메시지, 사용자 정보)
        """
        try:
            member_key = self._get_member_key(username)
            
            if not self._s3_key_exists(member_key):
                logger.warning(f"존재하지 않는 아이디로 로그인 시도: {username}")
                return False, '아이디 또는 비밀번호가 일치하지 않습니다.', None
                
            member_data = self._load_json_from_s3(member_key)
            if not member_data:
                logger.error(f"회원 데이터 로드 실패: {username}")
                return False, '로그인 처리 중 오류가 발생했습니다.', None
                
            if not member_data.get('is_active', True):
                logger.warning(f"비활성화된 계정 로그인 시도: {username}")
                return False, '사용할 수 없는 계정입니다.', None
                
            stored_password, salt = member_data['password'].rsplit(':', 1)
            hashed_password = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            ).hex()
            
            if hashed_password != stored_password:
                logger.warning(f"잘못된 비밀번호로 로그인 시도: {username}")
                return False, '아이디 또는 비밀번호가 일치하지 않습니다.', None
                
            # 마지막 로그인 시간 업데이트
            member_data['last_login'] = datetime.utcnow().isoformat()
            self._save_json_to_s3(member_key, member_data)
                
            logger.info(f"로그인 성공: {username}")
            return True, '로그인 성공', member_data
            
        except Exception as e:
            logger.error(f"로그인 중 오류 발생: {str(e)}")
            return False, '로그인 처리 중 오류가 발생했습니다.', None
            
    def get_user(self, username: str) -> Optional[Dict]:
        """사용자명으로 사용자 정보를 조회합니다."""
        try:
            member_key = self._get_member_key(username)
            return self._load_json_from_s3(member_key)
        except Exception as e:
            logger.error(f"사용자 정보 조회 중 오류: {str(e)}")
            return None

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
            member_key = self._get_member_key(username)
            if self._s3_key_exists(member_key):
                member_data = self._load_json_from_s3(member_key)
                if member_data:
                    member_data['last_login'] = datetime.utcnow().isoformat()
                    self._save_json_to_s3(member_key, member_data)
                    return True
            return False
        except Exception as e:
            logger.error(f"세션 업데이트 중 오류 발생: {str(e)}")
            return False

# 기존 코드와의 호환성을 위한 별칭
MemberManager = S3MemberManager
