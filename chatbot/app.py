import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import openai
import json
from pathlib import Path

app = FastAPI()

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 챗봇 클래스
class EmotionAwareChatbot:
    def __init__(self):
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
        4. 사용자의 감정을 잘 공유해주세요.
        5. 대답은 1-2문장으로 짧게 유지하세요.
        6. 영어 단어나 문장을 절대 사용하지 마세요.
        """

    def clean_text(self, text: str) -> str:
        """텍스트를 정제하는 함수"""
        try:
            return ''.join(c for c in text if ord(c) < 0x10000).strip()
        except Exception as e:
            print(f"텍스트 정제 중 오류 발생: {e}")
            return ""

    def save_conversation(self) -> None:
        """대화 기록을 JSON 파일로 저장"""
        try:
            if not self.chat_history:
                return
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_log/conversation_{timestamp}.json"
            os.makedirs("chat_log", exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
                
            print(f"대화 기록이 {filename}에 저장되었습니다.")
            
        except Exception as e:
            print(f"대화 기록 저장 중 오류 발생: {e}")
            
    def generate_response(self, user_input: str) -> str:
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # 최근 3개의 대화만 전달
            for entry in self.chat_history[-3:]:
                messages.append({"role": "user", "content": entry["user_input"]})
                messages.append({"role": "assistant", "content": entry["response"]})
            
            messages.append({"role": "user", "content": user_input})
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=100
            )
            
            response_text = response.choices[0].message.content.strip()
            cleaned_response = self.clean_text(response_text)
            
            # 대화 기록에 추가
            self.chat_history.append({
                "user_input": user_input,
                "response": cleaned_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return cleaned_response
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
            return "죄송합니다. 잠시 후 다시 시도해 주세요."

def load_chat_data() -> Dict[str, Any]:
    """여러 JSON 파일에서 대화 데이터를 로드합니다."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_files_dir = os.path.join(base_dir, "test_file")
        
        # 모든 JSON 파일 로드
        chat_data = {}
        for filename in os.listdir(test_files_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(test_files_dir, filename)
                print(f"Loading chat data from: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 중복 키가 있을 경우 나중에 로드된 파일의 값으로 덮어씁니다.
                    for category, emotions in data.items():
                        if category not in chat_data:
                            chat_data[category] = {}
                        for emotion, messages in emotions.items():
                            chat_data[category][emotion] = messages
                    
                    print(f"Successfully loaded {len(data)} categories from {filename}")
        
        print(f"Total loaded categories: {list(chat_data.keys())}")
        return chat_data
        
    except Exception as e:
        print(f"Error loading chat data: {str(e)}")
        return {}

# 전역 챗봇 인스턴스 및 대화 데이터 로드
chatbot = EmotionAwareChatbot()
chat_data = load_chat_data()

# 라우트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # JSON에서 카테고리 목록 추출 (예: ["청소년 여성 학교생활", "성인 남성 직장생활", ...])
    categories = list(chat_data.keys())
    return templates.TemplateResponse("index.html", {
        "request": request,
        "categories": categories
    })

@app.get("/api/chat-data")
async def get_chat_data():
    """챗봇 데이터를 JSON 형식으로 반환합니다."""
    return JSONResponse(content=chat_data)

@app.get("/api/start-auto-test")
async def start_auto_test(category: str, emotion: str):
    """자동 테스트를 시작합니다."""
    try:
        # 선택한 카테고리와 감정에 해당하는 메시지 목록 가져오기
        messages = chat_data.get(category, {}).get(emotion, [])
        if not messages:
            return JSONResponse(
                status_code=404,
                content={"error": f"'{category}' 카테고리의 '{emotion}' 감정에 해당하는 메시지를 찾을 수 없습니다."}
            )
        
        # 테스트에 필요한 정보 반환
        return {
            "success": True,
            "total_messages": len(messages),
            "messages": [msg["content"] for msg in messages]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"자동 테스트를 시작하는 중 오류가 발생했습니다: {str(e)}"}
        )

class Message(BaseModel):
    text: str
    is_user: bool

class ChatRequest(BaseModel):
    message: str
    action: str  # 'send' 또는 'end_conversation'


@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    if chat_request.action == "end_conversation":
        # 대화 종료 시 대화 기록 저장
        chatbot.save_conversation()
        # 대화 기록 초기화
        chatbot.chat_history = []
        return {"response": "대화가 종료되었습니다.", "end_conversation": True}
    
    # 일반 메시지 처리
    user_input = chat_request.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="메시지를 입력해주세요.")
    
    response = chatbot.generate_response(user_input)
    return {"response": response, "end_conversation": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
