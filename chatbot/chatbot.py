import json
import os
import unicodedata
import re
from datetime import datetime
from typing import Dict, List, Tuple

import openai

def clean_text(text: str) -> str:
    """
    텍스트를 UTF-8 인코딩에 안전한 형식으로 정제합니다.
    """
    try:
        # 모든 비표준 유니코드 제거
        text = ''.join(c for c in text if ord(c) < 0x10000)
        
        # UTF-8 인코딩/디코딩
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        
        # 제어 문자 제거
        text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
        
        # 빈 문자열 제거
        if not text.strip():
            return ""
        
        return text.strip()
    except Exception as e:
        print(f"텍스트 정제 중 오류 발생: {e}")
        return ""

class EmotionAwareChatbot:
    def __init__(self):
        """
        감정 인식 챗봇 클래스 (ChatGPT 기반)
        """
        self.emotion_history = []
        self.emotion_labels = ['기쁨', '분노', '불안', '슬픔', '혐오', '놀람']
        self.chat_history = []
        
        # API 키 설정
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")
            
        # OpenAI 클라이언트 초기화
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # 시스템 프롬프트 설정
        self.system_prompt = """
        당신은 친절하고 이해력이 뛰어난 상담사입니다.
        
        대화 규칙:
        1. 항상 한국어로만 대답하세요.
        2. 존댓말을 사용하세요.
        3. 간결하고 친절하게 답변하세요.
        4. 사용자의 감정을 잘 공감해주세요.
        5. 대답은 1-2문장으로 짧게 유지하세요.
        6. 영어 단어나 문장을 절대 사용하지 마세요.
        """

    def save_conversation_entry(self, user_input: str, response: str):
        """대화 기록 업데이트"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chat_history.append({
            "timestamp": timestamp,
            "user_input": user_input,
            "response": response
        })

    def save_conversation(self) -> None:
        """
        대화 기록을 JSON 파일로 저장합니다.
        """
        try:
            # 타임스탬프 기반 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_log/conversation_{timestamp}.json"
            
            # 디렉토리가 없으면 생성
            os.makedirs("chat_log", exist_ok=True)
            
            # 대화 기록 저장
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
                
            print(f"대화 기록이 {filename}에 저장되었습니다.")
            
        except Exception as e:
            print(f"대화 기록 저장 중 오류 발생: {e}")
            
    def generate_response(self, user_input: str) -> str:
        try:
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            for entry in self.chat_history[-3:]:
                messages.append({"role": "user", "content": entry["user_input"]})
                messages.append({"role": "assistant", "content": entry["response"]})
            
            messages.append({"role": "user", "content": user_input})
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100
                )
                
                response_text = response.choices[0].message.content.strip()
                cleaned_response = ''.join(c for c in response_text if ord(c) < 0x10000)
                cleaned_response = clean_text(cleaned_response)
                
                self.chat_history.append({
                    "user_input": user_input,
                    "response": cleaned_response
                })
                
                return cleaned_response
                
            except Exception as e:
                print(f"OpenAI API 호출 중 오류 발생: {e}")
                if self.chat_history:
                    last_response = self.chat_history[-1]["response"]
                    return last_response
                else:
                    return "죄송합니다. 잠시 후 다시 시도해 주세요."
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
            
            # 이전 대화 기록에서 마지막 응답을 가져와 재사용
            last_response = "죄송해요, 답변을 생성하는 데 잠시 문제가 생겼어요. 다시 시도해주시겠어요?"
            if self.chat_history:
                last_entry = self.chat_history[-1]
                last_response = last_entry["response"]
            
            return last_response, "중립", {"중립": 1.0}

    def get_emotion_summary(self) -> Dict[str, float]:
        """
        현재까지의 대화에서의 감정 분포를 반환합니다.
        
        Returns:
            Dict[str, float]: 감정별 점수 요약
        """
        # 영어 감정을 한국어로 매핑
        emotion_map = {
            'joy': '기쁨',
            'happiness': '기쁨',
            'anger': '분노',
            'fear': '불안',
            'sadness': '슬픔',
            'neutral': '중립',
            'disgust': '혐오',
            'surprise': '당황'
        }
        
        summary = {emotion: 0.0 for emotion in self.emotion_labels}
        
        for entry in self.emotion_history:
            for emotion, score in entry['emotion'].items():
                # 영어 감정을 한국어로 변환
                korean_emotion = emotion_map.get(emotion.lower(), emotion)
                # 한국어 감정이 유효한 경우에만 추가
                if korean_emotion in summary:
                    summary[korean_emotion] += score
                else:
                    # 유효하지 않은 감정은 '중립'으로 처리
                    summary['중립'] += score
                    
        # 정규화
        total = len(self.emotion_history)

def main():
    print("챗봇을 시작합니다...")
    print("종료하려면 '종료'를 입력하세요.")
    
    chatbot = EmotionAwareChatbot()
    
    try:
        while True:
            user_input = input("\n당신: ").strip()
            
            if user_input.lower() == "종료":
                print("\n대화 기록을 저장합니다...")
                chatbot.save_conversation()
                print("감사합니다! 챗봇을 종료합니다.")
                return
                
            try:
                response = chatbot.generate_response(user_input)
                print(f"\n봇: {response}")
                
                # 대화 기록 업데이트
                chatbot.save_conversation_entry(user_input, response)
                
            except Exception as e:
                print(f"\n봇: 죄송해요, 잠시 문제가 생겼어요. 다시 시도해주세요. ({str(e)})")
    except KeyboardInterrupt:
        print("\n대화 기록을 저장합니다...")
        chatbot.save_conversation()
        print("감사합니다! 챗봇을 종료합니다.")
        return

if __name__ == "__main__":
    main()
