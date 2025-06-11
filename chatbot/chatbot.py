import json
import re
import os
import requests
import datetime
import numpy as np
from typing import Dict, List, Tuple

# LM Studio 설정
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "local-model"  # LM Studio에서 로드한 모델 이름

class EmotionAwareChatbot:
    def __init__(self):
        """
        감정 인식 챗봇 클래스 (LM Studio 기반)
        
        Args:
            None
        """
        print("LM Studio 서버에 연결 중...")
        self.headers = {
            "Content-Type": "application/json"
        }
        try:
            # LM Studio 서버 상태 확인
            response = requests.get("http://localhost:1234/v1/models", headers=self.headers)
            response.raise_for_status()
            print(f"LM Studio 서버에 연결되었습니다. 사용 가능한 모델: {response.json()}")
            
        except requests.exceptions.RequestException as e:
            print(f"LM Studio 서버 연결 오류: {e}")
            print("LM Studio가 설치되어 있고 실행 중인지 확인해주세요.")
            print("LM Studio에서 한국어 모델을 로드한 후 API 서버를 실행해주세요.")
            print("API 서버 실행 방법: LM Studio에서 원하는 모델 로드 후 'Local Server' 탭에서 'Start Server' 클릭")
            raise
            
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            print("\n문제 해결을 위해 다음을 시도해 보세요:")
            print("1. 인터넷 연결 확인")
            print("2. GPU 메모리 확인 (최소 8GB 권장)")
            print("3. 더 작은 모델 사용 시도 (예: 'beomi/OPEN-SOLAR-KO-7B')")
            raise
        
        self.emotion_history = []
        self.emotion_labels = ['기쁨', '분노', '불안', '슬픔', '중립', '혐오', '당황']
        self.chat_history = []
        
        # 시스템 프롬프트 설정
        self.system_prompt = """당신은 공감과 배려를 바탕으로 한 상담사입니다. 
사용자의 감정을 이해하고 공감하며, 따뜻하고 친절한 어조로 대화하세요.
간단한 인사에는 적절히 답변하고, 감정이 담긴 메시지에는 공감과 위로를 전달하세요."""
        
        # 감정 분석을 위한 프롬프트
        self.emotion_prompt = """다음 사용자 메시지의 감정을 다음 7가지 중에서 하나로 분류해주세요: 
        기쁨, 분노, 불안, 슬픔, 중립, 혐오, 당황.
        응답은 반드시 다음 JSON 형식으로만 해주세요: {"emotion": "분석된_감정"}
        
        사용자 메시지: """
    
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """
        LM Studio를 사용하여 텍스트의 감정을 분석합니다.
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            Dict[str, float]: 감정 점수
        """
        prompt = """다음 사용자 메시지의 감정을 다음 7가지 중에서 하나로 분류해주세요: 
        기쁨, 분노, 불안, 슬픔, 중립, 혐오, 당황.
        
        반드시 다음 JSON 형식으로만 답변하세요:
        {"emotion": "감정", "confidence": 0.0}
        
        사용자 메시지: """ + text
        
        try:
            # LM Studio API를 통한 감정 분석
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "당신은 감정 분석 전문가입니다. 주어진 텍스트의 감정을 분석해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 더 일관된 응답을 위해 낮은 온도 사용
                max_tokens=100
            )
            
            response = completion.choices[0].message.content
            
            # JSON 형식으로 파싱 시도
            try:
                # 응답에서 JSON 부분만 추출
                if '{' in response and '}' in response:
                    json_str = response.split('{', 1)[1].rsplit('}', 1)[0]
                    json_str = '{' + json_str + '}'
                    result = json.loads(json_str)
                    emotion = result.get('emotion', '중립')
                    confidence = float(result.get('confidence', 1.0))
                    emotion_scores = {emotion: confidence}
                    dominant_emotion = emotion
                else:
                    raise ValueError("JSON 형식이 아닙니다.")
            except Exception as e:
                print(f"감정 분석 파싱 오류: {e}")
                emotion_scores = {'중립': 1.0}
                dominant_emotion = '중립'
            
            # 감정 기록에 추가
            self.emotion_history.append({
                'text': text,
                'emotion': emotion_scores,
                'dominant_emotion': dominant_emotion
            })
            
            return emotion_scores
            
        except Exception as e:
            print(f"감정 분석 중 오류 발생: {e}")
            return {e: 0.0 for e in self.emotion_labels}
    
    def generate_response(self, user_input: str) -> Tuple[str, str, Dict[str, float]]:
        """
        사용자 입력에 대한 응답 생성
        
        Args:
            user_input (str): 사용자 입력
            
        Returns:
            Tuple[str, str, Dict[str, float]]: (생성된 응답, 지배적인 감정, 감정 점수)
        """
        try:
            # 감정 분석
            emotion_scores = self.analyze_emotion(user_input)
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            # 대화 기록에 사용자 메시지 추가
            self.chat_history.append({"role": "user", "content": user_input})
            
            # 시스템 메시지와 대화 기록 준비
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # 최근 대화 3개만 사용 (메모리 절약을 위해)
            for msg in self.chat_history[-6:]:  # 최근 3턴(6개 메시지) 유지
                messages.append(msg)
            
            # LM Studio API에 전달할 메시지 형식 변환
            # 시스템 프롬프트 추가 (한국어 최적화)
            system_prompt = {
                "role": "system",
                "content": """
                당신은 친절하고 공감하는 한국어 상담사입니다. 사용자의 감정을 잘 이해하고 공감해주며, 
                따뜻하고 친절한 말투로 대화합니다. 존댓말을 사용하며, 사용자의 감정을 위로하고 격려합니다.
                """.strip()
            }
            
            # 시스템 프롬프트를 포함한 메시지 구성
            lm_messages = [system_prompt] + [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]
            
            try:
                # LM Studio API 호출
                payload = {
                    "model": MODEL_NAME,
                    "messages": lm_messages,
                    "temperature": 0.7,
                    "max_tokens": 200,
                    "stop": ["\n###"]
                }
                
                response = requests.post(
                    LMSTUDIO_API_URL,
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                # 응답 추출
                response_text = response.json()["choices"][0]["message"]["content"].strip()
                
                # 응답이 비어있는 경우 기본 응답 반환
                response = response_text if response_text else "죄송합니다. 이해하지 못했습니다. 다시 말씀해 주시겠어요?"
                    
            except Exception as e:
                print(f"LM Studio API 오류: {e}")
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    print(f"에러 응답: {e.response.text}")
                response = "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다."
            
            # 대화 기록에 챗봇 응답 추가
            self.chat_history.append({"role": "assistant", "content": response})
            
            # 대화 내용을 파일에 저장 (추가)
            self.save_conversation(user_input, response, dominant_emotion, emotion_scores)
            
            return response, dominant_emotion, emotion_scores
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
            return "죄송해요, 답변을 생성하는 데 잠시 문제가 생겼어요. 다시 시도해주시겠어요?", "중립", {"중립": 1.0}
    
    def save_conversation(self, user_input: str, bot_response: str, emotion: str, emotion_scores: Dict[str, float]):
        """대화 내용을 JSON 파일로 저장합니다."""
        try:
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "user_input": user_input,
                "bot_response": bot_response,
                "emotion": emotion,
                "emotion_scores": emotion_scores
            }
            
            # 대화 로그 파일에 추가
            with open("conversation_log.json", "a", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write("\n")  # 각 항목을 줄바꿈으로 구분
                
        except Exception as e:
            print(f"대화 저장 중 오류: {e}")
    
    def get_emotion_summary(self) -> Dict[str, float]:
        """
        현재까지의 대화에서의 감정 분포를 반환합니다.
        
        Returns:
            Dict[str, float]: 감정별 점수 요약
        """
        if not self.emotion_history:
            return {e: 0.0 for e in self.emotion_labels}
            
        summary = {e: 0.0 for e in self.emotion_labels}
        
        for entry in self.emotion_history:
            for emotion, score in entry['emotion'].items():
                summary[emotion] += score
                
        # 정규화
        total = len(self.emotion_history)
        for emotion in summary:
            summary[emotion] /= total
            
        return summary
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """
        가장 지배적인 감정을 반환합니다.
        
        Returns:
            Tuple[str, float]: (가장 지배적인 감정, 점수)
        """
        if not self.emotion_history:
            return ("대화 내역이 없습니다.", 0.0)
            
        summary = self.get_emotion_summary()
        dominant_emotion = max(summary.items(), key=lambda x: x[1])
        return dominant_emotion

def main():
    # 챗봇 초기화
    print("감정 인식 챗봇을 시작합니다...")
    print("종료하려면 '종료'를 입력하세요.")
    
    chatbot = EmotionAwareChatbot()
    
    while True:
        user_input = input("\n당신: ")
        
        if user_input.lower() in ['종료', '끝', 'quit', 'exit']:
            print("\n챗봇: 대화를 종료합니다. 감사합니다!")
            break
            
        response, emotion, scores = chatbot.generate_response(user_input)
        print(f"\n챗봇: {response}")
        print(f"[감정 분석] {emotion} (점수: {scores[emotion]:.2f})")
    
    # 대화 종료 시 감정 요약 출력
    summary = chatbot.get_emotion_summary()
    print("\n===== 대화 감정 요약 =====")
    for emotion, score in summary.items():
        print(f"{emotion}: {score*100:.1f}%")
    
    dominant_emotion, dominant_score = chatbot.get_dominant_emotion()
    print(f"\n가장 지배적인 감정: {dominant_emotion} ({dominant_score*100:.1f}%)")

if __name__ == "__main__":
    main()
