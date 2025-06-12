import json
import os
import sys
from typing import Dict, List, Optional, Any
from openai import OpenAI
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import settings and logger after adding to path
from chatbot.config import settings
from chatbot.config.logger import logger

class SimpleChatbot:
    """
    간단한 챗봇 클래스입니다.
    OpenAI의 GPT 모델을 사용하여 대화를 생성합니다.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        SimpleChatbot 초기화
        
        Args:
            api_key: OpenAI API 키. None인 경우 환경변수에서 가져옵니다.
            model: 사용할 OpenAI 모델명 (기본값: gpt-3.5-turbo)
        """
        self.chat_history: List[Dict[str, str]] = []
        self.model = model
        
        # API 키 설정
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다. "
                "환경변수 OPENAI_API_KEY를 설정하거나 생성자에 api_key를 전달하세요."
            )
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=self.api_key)
        
        # 시스템 프롬프트 설정
        self.system_prompt = """
        당신은 친절하고 전문적인 상담사입니다. 
        사용자의 질문에 도움이 되는 조언을 제공해주세요.
        
        대화 규칙:
        1. 항상 한국어로만 대답하세요.
        2. 존댓말을 사용하세요.
        3. 간결하고 친절하게 답변하세요.
        4. 대답은 1-2문장으로 짧게 유지하세요.
        5. 영어 단어나 문장을 절대 사용하지 마세요.
        """
        
        logger.info(f"SimpleChatbot initialized with model: {self.model}")
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        입력된 텍스트를 안전하게 정제합니다.
        
        Args:
            text: 정제할 텍스트
            
        Returns:
            str: 정제된 텍스트
            
        Raises:
            TypeError: 입력이 문자열, bytes, bytearray가 아닌 경우
        """
        if text is None:
            return ""
            
        try:
            # 문자열로 변환 (bytes나 bytearray인 경우)
            if not isinstance(text, str):
                text = text.decode('utf-8', errors='replace')
            
            # None 체크
            if text is None:
                return ""
                
        except Exception as e:
            # 변환 중 오류 발생 시 빈 문자열 반환
            print(f"Error cleaning text: {e}")
            return ""
            
        # 좌우 공백 제거
        return text.strip()
    
    def save_conversation_entry(self, user_input: str, response: str):
        """대화 기록을 업데이트합니다."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chat_history.append({
            "timestamp": timestamp,
            "user_input": user_input,
            "response": response
        })

    def save_conversation(self, filename: Optional[str] = None) -> str:
        """
        현재까지의 대화 기록을 JSON 파일로 저장합니다.
        
        Args:
            filename: 저장할 파일 경로. None인 경우 자동 생성됩니다.
            
        Returns:
            str: 저장된 파일 경로
            
        Raises:
            IOError: 파일 저장 중 오류가 발생한 경우
        """
        if not self.chat_history:
            logger.warning("저장할 대화 기록이 없습니다.")
            return ""
            
        try:
            import datetime
            # 파일명이 지정되지 않은 경우 자동 생성
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs(settings.CHAT_LOG_DIR, exist_ok=True)
                filename = os.path.join(settings.CHAT_LOG_DIR, f"conversation_{timestamp}.json")
            
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # 임시 파일에 먼저 저장 (원자적 연산을 위함)
            temp_filename = f"{filename}.tmp"
            with open(temp_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'created_at': datetime.datetime.utcnow().isoformat(),
                        'model': self.model
                    },
                    'conversation': self.chat_history
                }, f, ensure_ascii=False, indent=2)
            
            # 임시 파일을 최종 파일로 이동 (원자적 연산)
            if os.path.exists(filename):
                os.remove(filename)
            os.rename(temp_filename, filename)
            
            logger.info(f"대화 기록이 {filename}에 성공적으로 저장되었습니다.")
            return filename
            
        except (IOError, OSError, json.JSONEncodeError) as e:
            error_msg = f"대화 기록 저장 중 오류 발생: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise IOError(error_msg) from e
            
        except Exception as e:
            error_msg = f"예상치 못한 오류로 대화 기록을 저장하지 못했습니다: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
            
    def generate_response(
        self,
        user_input: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        사용자 입력에 대한 응답을 생성합니다.
        
        Args:
            user_input: 사용자 입력 텍스트
            max_tokens: 생성할 최대 토큰 수 (기본값: 500)
            temperature: 응답의 무작위성 (0.0 ~ 2.0, 기본값: 0.7)
            
        Returns:
            생성된 응답 텍스트
            
        Raises:
            ValueError: user_input이 비어있거나 유효하지 않은 경우
            Exception: API 오류 또는 기타 문제 발생 시
        """
        # 입력 검증
        if not user_input or not isinstance(user_input, str):
            raise ValueError("user_input은 비어있지 않은 문자열이어야 합니다.")
            
        try:
            # 입력 정제
            cleaned_input = self.clean_text(user_input)
            if not cleaned_input:
                return "죄송해요, 이해하지 못했어요. 다시 말씀해주시겠어요?"
            
            # API에 보낼 메시지 준비
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # 대화 기록 추가
            messages.extend([
                {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
                for i, msg in enumerate(self.chat_history[-10:])  # 최근 5턴(유저+봇)만 사용
            ])
            
            # 현재 사용자 입력 추가
            messages.append({"role": "user", "content": cleaned_input})
            
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=min(max(max_tokens, 1), 2000),  # 1~2000 사이로 제한
                temperature=min(max(temperature, 0.0), 2.0),  # 0.0 ~ 2.0 사이로 제한
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            
            # 응답 추출
            assistant_response = response.choices[0].message.content.strip()
            
            # 대화 기록 업데이트
            self.save_conversation_entry(cleaned_input, assistant_response)
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
            return "죄송해요, 답변을 생성하는 데 문제가 발생했어요. 잠시 후 다시 시도해주세요."

def main():
    """
    챗봇을 실행하는 메인 함수입니다.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Chatbot')
    parser.add_argument('--api-key', type=str, help='OpenAI API 키')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help=f'사용할 모델 (기본값: gpt-3.5-turbo)')
    
    args = parser.parse_args()
    
    try:
        # 챗봇 인스턴스 생성
        chatbot = SimpleChatbot(api_key=args.api_key, model=args.model)
        print("챗봇이 시작되었습니다. 종료하려면 '종료'를 입력하세요.")
        
        while True:
            try:
                # 사용자 입력 받기
                user_input = input("\n당신: ")
                
                # 종료 조건
                if user_input.lower() in ['종료', 'exit', 'quit']:
                    print("챗봇을 종료합니다. 안녕히 가세요!")
                    break
                
                # 응답 생성
                response = chatbot.generate_response(user_input)
                print(f"\n챗봇: {response}")
                
            except KeyboardInterrupt:
                print("\n챗봇을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류가 발생했습니다: {e}")
                continue
                
    except Exception as e:
        print(f"챗봇 초기화 중 오류가 발생했습니다: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
