import json
import hashlib
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from config.settings import settings
from config.logger import logger

class S3MemberManager:
    def __init__(self):
        self.bucket_name = 'kibwa-05'
        self.prefix = 'project/member_information/'
        
        # 환경 변수 확인
        aws_access_key_id = os.getenv('KIBWA05_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('KIBWA05_SECRET_ACCESS_KEY')
        region_name = os.getenv('KIBWA05_DEFAULT_REGION', 'ap-northeast-3')
        
        if not all([aws_access_key_id, aws_secret_access_key]):
            raise ValueError("S3 자격 증명 정보가 설정되지 않았습니다. KIBWA05_ACCESS_KEY_ID와 KIBWA05_SECRET_ACCESS_KEY를 확인하세요.")
            
        logger.info(f"Initializing S3MemberManager with bucket: {self.bucket_name}, region: {region_name}")
        
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

    def _get_member_key(self, username: str) -> str:
        """회원 정보 파일의 S3 키를 반환합니다."""
        return f"{self.prefix}{username}.json"

    def _hash_password(self, password: str) -> str:
        """비밀번호를 해시화합니다."""
        salt = secrets.token_hex(16)
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex() + f":{salt}"

    def _s3_key_exists(self, key: str) -> bool:
        """S3에 키가 존재하는지 확인합니다."""
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise

    def _load_json_from_s3(self, key: str) -> Optional[dict]:
        """S3에서 JSON 파일을 로드합니다."""
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    def _save_json_to_s3(self, key: str, data: dict):
        """JSON 데이터를 S3에 저장합니다."""
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'),
            ContentType='application/json'
        )

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
            member_key = self._get_member_key(username)
            
            if self._s3_key_exists(member_key):
                logger.warning(f"이미 존재하는 아이디로 가입 시도: {username}")
                return False, '이미 존재하는 아이디입니다.'
                
            hashed_password = self._hash_password(password)
            member_data = {
                'username': username,
                'password': hashed_password,
                'created_at': datetime.utcnow().isoformat(),
                'last_login': None,
                'is_active': True
            }
            
            self._save_json_to_s3(member_key, member_data)
            logger.info(f"새로운 회원 가입 성공: {username}")
            return True, '회원가입이 완료되었습니다.'
            
        except Exception as e:
            logger.error(f"회원가입 중 오류 발생: {str(e)}")
            return False, '회원가입 중 오류가 발생했습니다.'

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
