"""
Kibwa Chatbot - 한국어 감정 인식 챗봇

이 모듈은 사용자의 감정을 인식하고 공감하는 지능형 챗봇을 제공합니다.
OpenAI의 GPT 모델을 기반으로 하며, 사용자의 텍스트 입력에서 감정을 분석하여
상황에 맞는 공감 대화를 이어나갑니다.

주요 기능:
- 실시간 감정 분석
- 문맥을 고려한 자연스러운 대화
- 대화 기록 저장 및 관리
- 다양한 감정 상태에 따른 맞춤형 응답

사용 예시:
    >>> from chatbot import EmotionAwareChatbot
    >>> 
    >>> # 챗봇 인스턴스 생성 (API 키는 환경변수에 설정 필요)
    >>> chatbot = EmotionAwareChatbot()
    >>> 
    >>> # 사용자 입력에 대한 응답 생성
    >>> response = chatbot.generate_response("오늘 기분이 너무 좋아!")
    >>> print(response)
    "기분이 좋으시다니 다행이에요! 무슨 좋은 일이 있으셨나요?"
    >>> 
    >>> # 대화 기록 저장
    >>> chatbot.save_conversation()

모듈 구조:
- EmotionAwareChatbot: 감정 인식 챗봇의 메인 클래스
- config/: 설정 파일 및 로깅 설정
- utils/: 유틸리티 함수들
- tests/: 단위 테스트

의존성:
- openai>=1.0.0
- python-dotenv>=0.19.0
- pydantic>=1.8.0

라이센스:
MIT 라이센스
"""

from chatbot.chatbot import SimpleChatbot as EmotionAwareChatbot
from chatbot.config import settings
from chatbot.config.logger import logger

# 버전 정보
__version__ = "0.1.0"
__author__ = "Kibwa Team"
__license__ = "MIT"

# 공개 API
__all__ = [
    'EmotionAwareChatbot',
    'settings',
    '__version__',
    '__author__',
    '__license__'
]
