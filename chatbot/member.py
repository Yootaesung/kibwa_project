import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from config.settings import settings
from config.logger import logger

class MemberManager:
    """회원 정보를 관리하는 클래스입니다."""
    
    def __init__(self, member_dir: Optional[Path] = None):
        """
        MemberManager 초기화
        
        Args:
            member_dir: 회원 정보가 저장될 디렉토리 경로
        """
        self.member_dir = member_dir or settings.MEMBER_DIR
        self.member_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Member directory initialized at: {self.member_dir}")

    def _get_member_file(self, username: str) -> Path:
        """
        회원 정보 파일 경로를 반환합니다.
        
        Args:
            username: 사용자명
            
        Returns:
            Path: 회원 정보 파일 경로
        """
        if not username or not isinstance(username, str) or not username.strip():
            raise ValueError("유효하지 않은 사용자명입니다.")
            
        # 파일명에 안전한 문자열만 사용
        safe_username = "".join(c for c in username if c.isalnum() or c in ('_', '-', '.')).rstrip()
        return self.member_dir / f"{safe_username}.json"

    @staticmethod
    def _hash_password(password: str) -> str:
        """
        비밀번호를 해시화합니다.
        
        Args:
            password: 해시화할 비밀번호
            
        Returns:
            str: 해시화된 비밀번호
        """
        if not isinstance(password, str) or not password.strip():
            raise ValueError("비밀번호는 비어있을 수 없습니다.")
            
        salt = os.urandom(16).hex()
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex() + f":{salt}"

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
            member_file = self._get_member_file(username)
            
            if member_file.exists():
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
            
            with open(member_file, 'w', encoding='utf-8') as f:
                json.dump(member_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"새로운 회원 가입: {username}")
            return True, '회원가입이 완료되었습니다.'
            
        except Exception as e:
            logger.error(f"회원가입 중 오류 발생: {str(e)}", exc_info=True)
            return False, '회원가입 중 오류가 발생했습니다.'

    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """
        사용자 로그인을 처리합니다.
        
        Args:
            username: 사용자명
            password: 비밀번호
            
        Returns:
            Tuple[bool, str]: (성공 여부, 메시지)
        """
        try:
            member_file = self._get_member_file(username)
            
            if not member_file.exists():
                logger.warning(f"존재하지 않는 아이디로 로그인 시도: {username}")
                return False, '아이디 또는 비밀번호가 일치하지 않습니다.'
                
            with open(member_file, 'r', encoding='utf-8') as f:
                member_data = json.load(f)
            
            # 비밀번호 검증
            stored_password, salt = member_data['password'].rsplit(':', 1)
            hashed_password = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            ).hex()
            
            if hashed_password == stored_password:
                # 마지막 로그인 시간 업데이트
                member_data['last_login'] = datetime.utcnow().isoformat()
                with open(member_file, 'w', encoding='utf-8') as f:
                    json.dump(member_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"로그인 성공: {username}")
                return True, '로그인 성공'
            else:
                logger.warning(f"잘못된 비밀번호로 로그인 시도: {username}")
                return False, '아이디 또는 비밀번호가 일치하지 않습니다.'
                
        except Exception as e:
            logger.error(f"로그인 중 오류 발생: {str(e)}", exc_info=True)
            return False, '로그인 중 오류가 발생했습니다.'

    def check_member(self, username):
        """회원 존재 여부 확인"""
        member_file = self._get_member_file(username)
        return os.path.exists(member_file)

    def update_session(self, username):
        """세션 업데이트"""
        member_file = self._get_member_file(username)
        if os.path.exists(member_file):
            with open(member_file, 'r', encoding='utf-8') as f:
                member_data = json.load(f)
            member_data['last_login'] = datetime.now().isoformat()
            with open(member_file, 'w', encoding='utf-8') as f:
                json.dump(member_data, f, ensure_ascii=False, indent=2)
            return True
        return False
        
    def get_user(self, username):
        """사용자 정보 조회"""
        member_file = self._get_member_file(username)
        if not os.path.exists(member_file):
            return None
            
        with open(member_file, 'r', encoding='utf-8') as f:
            member_data = json.load(f)
            
        # 민감한 정보는 제외하고 반환
        return {
            'username': member_data.get('username'),
            'created_at': member_data.get('created_at'),
            'last_login': member_data.get('last_login')
        }
