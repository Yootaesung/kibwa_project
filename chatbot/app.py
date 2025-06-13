import os
import json
import hashlib
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any

import boto3
from fastapi import FastAPI, Request, HTTPException, Response, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional
import uuid
import os
from pathlib import Path

# ---------------------------
# 1. 환경설정 및 유틸리티
# ---------------------------
from dotenv import load_dotenv
import os
from pathlib import Path

# 환경 변수 로드 (절대 경로로 지정)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

# 환경 변수 확인
print("AWS_ACCESS_KEY_ID:", 'Set' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not Set')
print("AWS_SECRET_ACCESS_KEY:", 'Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not Set')
print("AWS_DEFAULT_REGION:", os.getenv('AWS_DEFAULT_REGION', 'Not Set'))

# 기본 설정
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 로그 디렉토리 생성
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'logs', 'error.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ---------------------------
# 2. FastAPI 인스턴스 및 미들웨어
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 정적 파일/템플릿 등록
# 정적 파일 설정
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 감정 데이터 디렉토리 설정
EMOTION_DATA_DIR = os.path.join(BASE_DIR, 'emotion_data')
os.makedirs(EMOTION_DATA_DIR, exist_ok=True)

# 감정 데이터 정적 파일 마운트
app.mount("/emotion_data", StaticFiles(directory=EMOTION_DATA_DIR), name="emotion_data")

# 템플릿 설정
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---------------------------
# 3. 멤버/채팅 관리 클래스
# ---------------------------
class MemberManager:
    def __init__(self):
        self.member_dir = os.path.join(BASE_DIR, 'member_information')
        os.makedirs(self.member_dir, exist_ok=True)

    def _get_member_file(self, username):
        return os.path.join(self.member_dir, f'{username}.json')

    def _hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, username, password):
        member_file = self._get_member_file(username)
        if os.path.exists(member_file):
            return False, "이미 존재하는 사용자입니다."
        hashed_password = self._hash_password(password)
        user_data = {
            'username': username,
            'password': hashed_password,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        try:
            with open(member_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            return True, "회원가입이 완료되었습니다."
        except Exception as e:
            return False, f"회원가입 중 오류가 발생했습니다: {str(e)}"

    def login(self, username, password):
        member_file = self._get_member_file(username)
        if not os.path.exists(member_file):
            return False, "사용자를 찾을 수 없습니다."
        try:
            with open(member_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            if user_data['password'] == self._hash_password(password):
                return True, user_data
            else:
                return False, "비밀번호가 일치하지 않습니다."
        except Exception as e:
            return False, f"로그인 중 오류가 발생했습니다: {str(e)}"

    def get_user(self, user_id):
        member_file = self._get_member_file(user_id)
        if not os.path.exists(member_file):
            return None
        try:
            with open(member_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

member_manager = MemberManager()

def get_user_chat_log_path(username: str):
    chat_log_dir = os.path.join(BASE_DIR, 'chat_logs')
    os.makedirs(chat_log_dir, exist_ok=True)
    return os.path.join(chat_log_dir, f"{username}_chat_log.json")

def load_chat_history(username: str) -> List[Dict[str, Any]]:
    log_file = get_user_chat_log_path(username)
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in chat log: {log_file}")
            return []
    return []

def save_chat_message(username: str, role: str, content: str):
    try:
        chat_log_path = get_user_chat_log_path(username)
        chat_history = []
        if os.path.exists(chat_log_path):
            try:
                with open(chat_log_path, 'r', encoding='utf-8') as f:
                    chat_history = json.load(f)
            except json.JSONDecodeError:
                chat_history = []
        chat_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        temp_path = f"{chat_log_path}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, chat_log_path)
    except Exception as e:
        logger.error(f"채팅 메시지 저장 중 오류 발생: {str(e)}")
        raise

def get_chat_context(username: str) -> List[Dict[str, str]]:
    return load_chat_history(username)

# ---------------------------
# 4. 챗봇 클래스
# ---------------------------
class SimpleChatbot:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")
        self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = (
            "당신은 친절하고 이해력이 뛰어난 상담사입니다.\n"
            "대화 규칙:\n"
            "1. 항상 한국어로만 대답하세요.\n"
            "2. 존댓말을 사용하세요.\n"
            "3. 간결하고 친절하게 답변하세요.\n"
            "4. 사용자의 감정을 잘 공유해주세요.\n"
            "5. 대답은 1-2문장으로 짧게 유지하세요.\n"
            "6. 영어 단어나 문장을 절대 사용하지 마세요.\n"
        )

    def generate_response(self, user_input: str, username: str = None, chat_history: list = None) -> str:
        try:
            if chat_history is None:
                chat_history = []
            chat_history.append({"role": "user", "content": user_input})
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(chat_history)
            total_tokens = sum(len(msg["content"].split()) for msg in messages)
            while total_tokens > 3000 and len(messages) > 1:
                messages.pop(1)
                total_tokens = sum(len(msg["content"].split()) for msg in messages)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            bot_response = response.choices[0].message.content
            return bot_response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "죄송합니다. 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

chatbot = SimpleChatbot()

# ---------------------------
# 5. 모델
# ---------------------------
class ChatRequest(BaseModel):
    message: str
    action: str
    is_filtered: bool = False
    is_test: bool = False
    emotion: str = "기쁨"  # 기본값으로 '기쁨' 설정

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class EndChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# ---------------------------
# 5.1 S3 클라이언트 설정
# ---------------------------
class S3Client:
    def __init__(self):
        # 환경 변수에서 자격 증명 가져오기
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region_name = os.getenv('AWS_DEFAULT_REGION', 'ap-southeast-2')
        
        print(f"Initializing S3 client with:\nAccess Key: {aws_access_key_id[:4]}...\nRegion: {region_name}")
        
        # S3 클라이언트 초기화
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.bucket_name = 'kibwa-12'
        self.prefix = 'project/'
    
    def list_scenarios(self):
        response = self.s3.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=self.prefix
        )
        
        scenarios = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json') and key != self.prefix:
                # 파일명에서 메타데이터 추출 (예: project/10대_남성_학생_1.json)
                filename = key.replace(self.prefix, '').replace('.json', '')
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    age_group = parts[0]
                    gender = parts[1]
                    role = '_'.join(parts[2:])  # 역할에 밑줄이 포함될 수 있음
                    
                    scenarios.append({
                        'key': key,
                        'filename': filename,
                        'age_group': age_group,
                        'gender': gender,
                        'role': role
                    })
        
        return scenarios
    
    def get_scenario(self, key: str):
        try:
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)
            
            # 다양한 JSON 형식 처리
            if isinstance(data, dict):
                # {감정: [...]} 형식인 경우
                if any(emotion in data for emotion in ['분노', '기쁨', '슬픔', '두려움', '놀람']):
                    messages = []
                    for emotion_list in data.values():
                        if isinstance(emotion_list, list):
                            for item in emotion_list:
                                if isinstance(item, dict) and all(field in item for field in 
                                    ['age_group', 'gender', 'role', 'situation', 'emotion', 'content']):
                                    messages.append(item)
                    return messages
                # 단일 객체인 경우
                elif all(field in data for field in 
                        ['age_group', 'gender', 'role', 'situation', 'emotion', 'content']):
                    return [data]
                else:
                    print(f"Invalid data format in scenario {key}")
                    return []
            # 리스트 형식인 경우
            elif isinstance(data, list):
                # 모든 항목이 올바른 형식인지 확인
                if all(isinstance(item, dict) and 
                      all(field in item for field in 
                          ['age_group', 'gender', 'role', 'situation', 'emotion', 'content'])
                      for item in data):
                    return data
                else:
                    # 일부 항목만 올바른 형식인 경우 필터링
                    valid_items = [item for item in data 
                                if isinstance(item, dict) and 
                                all(field in item for field in 
                                    ['age_group', 'gender', 'role', 'situation', 'emotion', 'content'])]
                    if valid_items:
                        print(f"Filtered out {len(data) - len(valid_items)} invalid items from {key}")
                        return valid_items
                    return []
            else:
                print(f"Unsupported data format in scenario {key}")
                return []
                
        except Exception as e:
            print(f"Error getting scenario {key}: {str(e)}")
            return []

s3_client = S3Client()

# ---------------------------
# 5.2 테스트 결과 저장 경로
# ---------------------------
TEST_RESULTS_DIR = os.path.join(BASE_DIR, 'test_chat_logs')
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

def save_test_result(scenario_key: str, messages: list) -> str:
    """테스트 결과를 파일로 저장하고 파일 경로를 반환합니다."""
    # 파일명 생성 (중복 방지를 위해 타임스탬프 추가)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{os.path.basename(scenario_key).replace('.json', '')}_{timestamp}.json"
    filepath = os.path.join(TEST_RESULTS_DIR, filename)
    
    # 디렉토리 생성 (필요한 경우)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 파일 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'scenario_key': scenario_key,
            'timestamp': datetime.now().isoformat(),
            'messages': messages
        }, f, ensure_ascii=False, indent=2)
    
    return filepath

# ---------------------------
# 6. 라우트
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return RedirectResponse(url="/login")

@app.get("/chat")
async def chat_page(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("index.html", {"request": request, "categories": ["일상", "업무", "학습", "여행", "음식"]})

@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    try:
        username = request.cookies.get('user_id')
        if not username:
            raise HTTPException(status_code=401, detail="Not authenticated")
        user_message = chat_request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # 테스트 모드일 때는 챗봇 응답을 생략하고 빈 문자열 반환
        if chat_request.is_test:
            if chat_request.action == 'test':
                return JSONResponse(content={"response": "", "status": "success"})
            else:
                raise HTTPException(status_code=400, detail="Invalid action for test mode")
        
        chat_context = get_chat_context(username)
        
        print(f"Received message: {user_message}")  # 디버깅용 로그
        
        # 챗봇 응답 생성
        response = chatbot.generate_response(
            user_input=user_message,
            username=username,
            chat_history=chat_context
        )
        print(f"Generated response: {response}")  # 디버깅용 로그
        
        # 메시지 저장
        save_chat_message(username, "assistant", response)
        
        return JSONResponse(content={"response": response, "status": "success"})
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # 디버깅용 로그
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/login")
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/test")
async def test_page(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})

# 테스트 관련 API 엔드포인트
@app.get("/api/test/scenarios")
async def get_test_scenarios():
    """S3 버킷에서 테스트 시나리오 목록을 가져옵니다."""
    try:
        scenarios = s3_client.list_scenarios()
        return {"scenarios": scenarios}
    except Exception as e:
        print(f"Error listing scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test/scenario/{scenario_key:path}")
async def get_scenario(scenario_key: str):
    """특정 시나리오의 상세 내용을 가져옵니다."""
    try:
        messages = s3_client.get_scenario(scenario_key)
        print(f"Loaded scenario data: {messages}")  # 디버깅용 로그 추가
        
        if not messages:
            raise HTTPException(status_code=404, detail="시나리오를 찾을 수 없거나 유효하지 않은 형식입니다.")
        
        # 각 메시지의 필수 필드 확인 및 기본값 설정
        for msg in messages:
            msg['content'] = msg.get('content', '')
            msg['emotion'] = msg.get('emotion', '중립')
        
        # 시나리오 데이터 구성
        scenario = {
            'key': scenario_key,
            'messages': messages
        }
        
        return {"scenario": scenario}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting scenario {scenario_key}: {e}")
        raise HTTPException(status_code=500, detail=f"시나리오를 처리하는 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/test/save")
async def save_test_result_endpoint(data: dict):
    """테스트 결과를 저장합니다."""
    try:
        scenario_key = data.get('scenario_key')
        messages = data.get('messages', [])
        
        if not scenario_key:
            raise HTTPException(status_code=400, detail="시나리오 키가 필요합니다.")
        
        # 테스트 결과 저장
        filepath = save_test_result(scenario_key, messages)
        
        return {"status": "success", "filepath": filepath}
    except Exception as e:
        print(f"Error saving test result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/register")
async def register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/api/login")
async def api_login(login_data: LoginRequest, response: Response):
    success, message = member_manager.login(login_data.username, login_data.password)
    if success:
        response_data = {"message": "로그인 성공", "redirect": "/chat"}
        response = JSONResponse(content=response_data)
        response.set_cookie(
            key="user_id",
            value=login_data.username,
            httponly=True,
            max_age=3600,
            samesite='lax',
            secure=False
        )
        return response
    return JSONResponse(content={"message": message}, status_code=401)

@app.post("/api/register")
async def api_register(register_data: RegisterRequest):
    if not register_data.username or not register_data.password:
        return JSONResponse(
            content={"message": "아이디와 비밀번호를 모두 입력해주세요."},
            status_code=400
        )
    if len(register_data.password) < 4:
        return JSONResponse(
            content={"message": "비밀번호는 최소 4자 이상이어야 합니다."},
            status_code=400
        )
    success, message = member_manager.register(register_data.username, register_data.password)
    if success:
        return JSONResponse(content={"message": message}, status_code=201)
    else:
        return JSONResponse(content={"message": message}, status_code=400)

@app.post("/api/end-chat")
async def end_chat(request: EndChatRequest, req: Request):
    username = req.cookies.get("user_id")
    if not username:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    for msg in request.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            save_chat_message(username, role, content)
    return {"message": "채팅 기록이 저장되었습니다."}

@app.get("/api/check-auth")
async def check_auth(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = member_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return JSONResponse(content={"username": user["username"], "message": "인증 성공"})

@app.get("/api/logout")
def logout(request: Request):
    response = RedirectResponse(url='/login')
    response.delete_cookie(key="user_id")
    return response

@app.get("/api/chat-data")
async def get_chat_data():
    # 필요시 실제 데이터로 교체
    return JSONResponse({"categories": ["일상", "업무", "학습", "여행", "음식"]})

@app.post("/api/chat")
async def chat(chat_request: ChatRequest, request: Request):
    # 테스트 모드인 경우 감정별 응답 반환
    if chat_request.is_test:
        import random
        
        # 감정별 응답 목록
        emotion_responses = {
            "기쁨": [
                "정말 기쁜 일이시군요! 기분이 좋아 보이네요 😊",
                "기쁜 일이 있으셨다니 다행이에요! 더 자세히 들려주실 수 있나요?",
                "즐거운 일이 있으셨군요! 기분이 좋아지는 대화네요.",
                "기쁜 마음이 전해져요! 계속해서 이야기해 주세요.",
                "행복한 일이 있으셨군요! 더 자세히 알려주세요.",
                "웃음이 가득한 하루가 되셨네요! 😄",
                "기쁨이 느껴지는 대화예요! 더 들려주실 수 있나요?",
                "행복한 에너지가 느껴져요! 기분이 좋아지네요.",
                "당신의 기쁨이 제게도 전해져요! 😊",
                "기쁜 소식이 있으셨나 봐요! 자세히 들려주세요.",
                "웃음이 멈추지 않으시네요! 무슨 일이신가요?",
                "행복한 순간을 함께 나눠주셔서 감사해요!"
            ],
            "분노": [
                "화가 나시는 마음, 충분히 이해해요. 속 시원히 털어놓으세요 😠",
                "정말 화가 나시겠어요. 더 자세히 말씀해 주시겠어요?",
                "화가 날 만한 일이셨군요. 감정을 표현해 주셔서 감사합니다.",
                "속상한 마음이 느껴져요. 더 말씀해 주실 수 있나요?",
                "분노를 느끼시는 게 당연하세요. 계속 이야기해 주세요.",
                "화가 나실 만한 상황이시군요. 마음껏 표현해 주세요. 💢",
                "분노가 느껴지네요. 제가 도울 수 있는 방법이 있을까요?",
                "화가 나는 감정을 표현해 주셔서 감사해요. 계속 말씀해 주세요.",
                "정말 속상하셨겠어요. 제가 잘 듣고 있어요.",
                "분노를 느끼는 건 자연스러운 일이에요. 마음껏 털어놓으세요.",
                "화가 나는 일이 있으셨군요. 같이 해결 방법을 찾아볼까요?",
                "당신의 감정에 공감해요. 마음껏 말씀해 주세요."
            ],
            "슬픔": [
                "마음이 아프시겠어요. 제가 여기 있어요 😢",
                "슬픈 일이 있으셨군요. 말씀해 주셔서 감사합니다.",
                "마음이 무거우시겠어요. 더 자세히 나눠보실래요?",
                "슬픔을 느끼고 계시군요. 제가 도울 수 있는 게 있을까요?",
                "마음이 아픈 일이 있으셨군요. 이야기해 주셔서 감사합니다.",
                "슬픔이 느껴지는 목소리예요. 제가 여기서 듣고 있을게요. 💔",
                "마음이 아프실 것 같아요. 조금씩 말씀해 보실래요?",
                "슬픔은 나누면 반이 된다고 하잖아요. 제가 함께 할게요.",
                "눈물을 흘리셔도 괜찮아요. 제가 여기 있어요.",
                "마음이 무겁게 느껴지시나 봐요. 함께 이야기해 볼까요?",
                "슬픔을 느끼는 건 당연한 일이에요. 제가 지켜보고 있을게요.",
                "당신의 아픔을 이해하려 노력할게요. 계속 말씀해 주세요."
            ],
            "두려움": [
                "불안하시겠어요. 안전하시다니 다행이에요 😨",
                "두려우셨겠어요. 더 자세히 말씀해 주실 수 있나요?",
                "불안한 마음이 느껴져요. 제가 도울 수 있는 게 있을까요?",
                "두려움을 느끼시는 게 당연해요. 계속 이야기해 주세요.",
                "불안한 마음이 드시는군요. 더 편안하게 말씀해 주세요.",
                "무서운 일이 있으셨군요. 제가 여기서 지켜보고 있을게요. 🛡️",
                "불안한 마음이 드시는군요. 함께 해결해 나가 볼까요?",
                "두려움을 느끼는 건 자연스러운 일이에요. 안전하다고 말씀드릴게요.",
                "불안할 때는 마음껏 이야기해 주세요. 제가 듣고 있을게요.",
                "두려움을 털어놓으시면 조금은 나아지실 거예요.",
                "안전한 공간이에요. 마음껏 두려움을 표현해 주세요.",
                "제가 옆에서 지켜보고 있을게요. 안심하세요."
            ],
            "놀람": [
                "놀라우셨겠어요! 어떤 일이 있었는지 더 들려주실 수 있나요? 😲",
                "깜짝 놀라셨겠어요! 더 자세한 이야기 해주실래요?",
                "예상치 못한 일이셨군요! 어떤 기분이 드시나요?",
                "놀라운 일이 있으셨군요! 더 말씀해 주세요.",
                "깜짝 놀라셨을 것 같아요. 계속 이야기해 주실 수 있나요?",
                "놀라운 일이 있으셨군요! 더 자세히 들려주세요! 🤯",
                "깜짝 놀라셨을 것 같아요. 무슨 일이 있었는지 말씀해 주실래요?",
                "예상치 못한 일이시군요! 어떤 기분이 드시나요?",
                "놀라운 소식이 있으셨나 봐요! 자세히 알려주세요.",
                "깜짝 놀라셨을 것 같아요. 괜찮으신가요?",
                "놀라운 일이 있으셨군요! 제가 도울 수 있는 게 있나요?",
                "예상치 못한 일이시군요! 더 자세히 이야기해 주실 수 있나요?"
            ],
            "혐오": [
                "불쾌하셨겠어요. 더 자세히 말씀해 주실 수 있나요? 🤢",
                "불편하신 마음이 느껴져요. 이야기해 주셔서 감사합니다.",
                "불쾌한 경험이셨군요. 더 자세히 나눠보실래요?",
                "혐오스러운 일이 있으셨군요. 제가 도울 수 있는 게 있을까요?",
                "불편한 감정이 드시는군요. 더 편하게 말씀해 주세요.",
                "불쾌한 일이 있으셨군요. 마음껏 털어놓으세요. 🚫",
                "불편한 감정이 드시는 것 같아요. 더 자세히 말씀해 주실 수 있나요?",
                "혐오스러운 경험이셨군요. 제가 여기서 듣고 있을게요.",
                "불쾌한 일이 있으셨다니 안타깝네요. 이야기해 주셔서 감사합니다.",
                "혐오스러운 상황이셨군요. 함께 해결 방법을 찾아볼까요?",
                "불편한 감정을 표현해 주셔서 감사해요. 계속 말씀해 주세요.",
                "불쾌한 경험이셨을 것 같아요. 제가 도울 수 있는 게 있을까요?"
            ]
        }
        
        # 현재 감정 가져오기
        current_emotion = chat_request.emotion
        
        # 감정에 맞는 응답 목록 가져오기 (기본값: 기쁨의 응답)
        responses = emotion_responses.get(current_emotion, emotion_responses["기쁨"])
        
        # 랜덤하게 응답 선택
        response = random.choice(responses)
        
        return JSONResponse(content={"response": response, "status": "success", "emotion": current_emotion})
    
    # 일반 채팅 모드
    username = request.cookies.get('user_id')
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    user_message = chat_request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # 욕설 필터링 기능 필요시 구현
    chat_context = get_chat_context(username)
    save_chat_message(username, "user", user_message)
    
    response = chatbot.generate_response(
        user_input=user_message,
        username=username,
        chat_history=chat_context
    )
    
    save_chat_message(username, "assistant", response)
    return {"response": response, "status": "success"}
