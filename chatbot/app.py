import os
import json
import hashlib
import logging
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import boto3
import botocore
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException, Form, status, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# OpenAI 클라이언트 초기화
openai_client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

# 챗봇 인스턴스 초기화
class ChatBot:
    def __init__(self):
        self.client = openai_client

    async def generate_response(self, message: str, username: str, chat_history: list):
        """채팅 응답을 생성합니다."""
        try:
            # 채팅 히스토리에 새로운 메시지 추가
            messages = [
                {"role": "system", "content": "당신은 친절한 챗봇입니다."}
            ] + chat_history + [
                {"role": "user", "content": message}
            ]

            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )

            # 응답 처리
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API 호출 실패: {str(e)}")
            raise Exception("채팅 처리 중 오류가 발생했습니다.")

# 챗봇 인스턴스 생성
chatbot = ChatBot()

# ---------------------------
# 1. 환경설정 및 유틸리티
# ---------------------------

# GitHub Actions 환경 변수 확인
print("KIBWA05_ACCESS_KEY_ID:", 'Set' if os.getenv('KIBWA05_ACCESS_KEY_ID') else 'Not Set')
print("KIBWA05_SECRET_ACCESS_KEY:", 'Set' if os.getenv('KIBWA05_SECRET_ACCESS_KEY') else 'Not Set')
print("KIBWA05_DEFAULT_REGION:", os.getenv('KIBWA05_DEFAULT_REGION', 'ap-northeast-3'))
print("AWS_ACCESS_KEY_ID:", 'Set' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not Set')
print("AWS_SECRET_ACCESS_KEY:", 'Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not Set')
print("AWS_DEFAULT_REGION:", os.getenv('AWS_DEFAULT_REGION'))

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

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000", 
        "http://3.107.174.223:8000",
        "http://3.107.174.223"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 인증 미들웨어
@app.middleware("http")
async def verify_auth(request: Request, call_next):
    # 인증이 필요 없는 경로
    no_auth_paths = [
        "/login", 
        "/register", 
        "/api/login", 
        "/api/register", 
        "/test",
        "/test/login",
        "/api/test/scenarios"
    ]
    if request.url.path in no_auth_paths or request.url.path.startswith('/static/'):
        return await call_next(request)
    
    # 채팅 페이지나 API 요청일 때 인증 확인
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    # MongoDB에서 사용자 검증
    user = member_manager.get_user(user_id)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    response = await call_next(request)
    return response

# 정적 파일/템플릿 등록
# 정적 파일 설정
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 감정 키워드 파일 추가
app.mount("/emotion_data", StaticFiles(directory="emotion_data"), name="emotion_data")

# S3 버킷 설정
TEST_BUCKET = 'kibwa-12'
TEST_PREFIX = 'project/'
CHATBOT_BUCKET = 'kibwa-05'
CHATBOT_PREFIX = 'project/'

# S3 클라이언트 초기화 (테스트 버킷용)
try:
    test_s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-3')
    )
    logger.info("Test S3 client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize test S3 client: {str(e)}")
    test_s3_client = None

# S3 클라이언트 초기화 (챗봇 버킷용)
try:
    chatbot_s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),  # KIBWA05_ACCESS_KEY_ID 대신 AWS_ACCESS_KEY_ID 사용
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),  # KIBWA05_SECRET_ACCESS_KEY 대신 AWS_SECRET_ACCESS_KEY 사용
        region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-3')
    )
    logger.info("Chatbot S3 client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot S3 client: {str(e)}")
    chatbot_s3_client = None

class S3Client:
    def __init__(self, bucket_type='chatbot'):
        """
        S3 클라이언트 초기화
        :param bucket_type: 'test' 또는 'chatbot' 중 하나
        """
        if bucket_type == 'test':
            self.bucket = TEST_BUCKET
            self.bucket_name = TEST_BUCKET
            self.prefix = TEST_PREFIX
            self.client = test_s3_client
        else:  # 'chatbot'
            self.bucket = CHATBOT_BUCKET
            self.bucket_name = CHATBOT_BUCKET
            self.prefix = CHATBOT_PREFIX
            self.client = chatbot_s3_client
    
    def get_file(self, key):
        """S3에서 파일을 가져옵니다."""
        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=key
            )
            content = response['Body'].read().decode('utf-8')
            
            # JSON 파일인 경우 파싱하여 반환
            if key.endswith('.json'):
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 오류: {str(e)}")
                    logger.error(f"파일 내용: {content}")
                    raise Exception(f"JSON 파일을 파싱하는 중 오류가 발생했습니다: {str(e)}")
            
            return content
        except Exception as e:
            logger.error(f"S3 get_object error: {str(e)}")
            raise Exception(f"S3에서 파일을 가져오는 중 오류가 발생했습니다: {str(e)}")

    def get_scenario(self, scenario_key):
        """특정 시나리오를 가져옵니다."""
        try:
            # S3에서 파일 내용 가져오기
            content = self.get_file(scenario_key)
            
            # JSON 데이터를 messages 배열로 변환
            messages = []
            for item in content:
                messages.append({
                    'role': 'user',
                    'content': item['content'],
                    'emotion': item['emotion']
                })
            
            return {"scenario": {"messages": messages}}
            
        except Exception as e:
            logger.error(f"S3에서 시나리오를 가져오는 중 오류 발생: {str(e)}")
            raise Exception(f"시나리오를 가져오는 중 오류가 발생했습니다: {str(e)}")

    def list_scenarios(self):
        """S3에서 테스트 시나리오 목록을 가져옵니다."""
        try:
            # JSON 파일 목록 가져오기
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.prefix,
                Delimiter='/'
            )
            
            # JSON 파일만 필터링
            scenarios = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # 키가 prefix로 시작하고 JSON 파일인 경우만 처리
                    if obj['Key'].startswith(self.prefix) and obj['Key'].endswith('.json'):
                        # prefix 제거 후 파일명에서 .json 확장자 제거하고, _를 공백으로 변경
                        relative_key = obj['Key'][len(self.prefix):]
                        name = os.path.basename(relative_key).replace('.json', '').replace('_', ' ')
                        scenarios.append({
                            'key': obj['Key'],
                            'name': name
                        })
            
            return scenarios
            
        except Exception as e:
            logger.error(f"S3 시나리오 목록 가져오기 실패: {str(e)}")
            raise Exception(f"S3에서 시나리오 목록을 가져오는 중 오류가 발생했습니다: {str(e)}")
    


def ensure_s3_paths():
    """필요한 S3 경로가 존재하는지 확인하고 없으면 생성"""
    required_paths = [
        'chat_logs/',
        'test_chat_logs/',
        'member_information/',
        'emotion_data/',
        'profanity_data/'
    ]
    
    try:
        # 버킷의 루트에 접근 가능한지 확인
        chatbot_s3_client.head_bucket(Bucket=CHATBOT_BUCKET)
        
        # 필요한 경로들이 있는지 확인하고 없으면 생성
        for path in required_paths:
            try:
                chatbot_s3_client.head_object(Bucket=CHATBOT_BUCKET, Key=f"{CHATBOT_PREFIX}{path}")
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == '404':
                    # 경로가 없으면 빈 디렉토리 생성
                    chatbot_s3_client.put_object(Bucket=CHATBOT_BUCKET, Key=f"{CHATBOT_PREFIX}{path}")
                else:
                    logger.error(f"S3 객체 접근 오류: {e}")
                    raise
    except Exception as e:
        logger.error(f"S3 버킷 접근 오류: {e}")
        raise

# S3 경로 확인
ensure_s3_paths()

# 로컬 디렉토리 생성 (임시 파일 처리용)
LOCAL_TEMP_DIR = os.path.join(BASE_DIR, 'temp')
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

# 템플릿 설정
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---------------------------
# 3. S3 멤버 관리 클래스
# ---------------------------
from chatbot.member import MemberManager, MongoDBManager

# MongoDB 매니저 인스턴스 생성
db_manager = MongoDBManager()

# MemberManager 인스턴스 생성
member_manager = MemberManager()

def get_user_chat_log_path(username: str):
    """사용자 채팅 로그의 S3 경로 반환"""
    return f"{CHATBOT_PREFIX}chat_logs/{username}_chat_log.json"

def load_chat_history(username: str) -> List[Dict[str, Any]]:
    """S3에서 채팅 기록 로드"""
    log_key = get_user_chat_log_path(username)
    try:
        response = chatbot_s3_client.get_object(Bucket=CHATBOT_BUCKET, Key=log_key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return []
        else:
            logger.error(f"S3 객체 접근 오류: {e}")
            return []
    except Exception as e:
        logger.error(f"채팅 기록 로드 중 오류: {e}")
        return []

def save_chat_message(username: str, role: str, content: str):
    """채팅 메시지를 S3에 저장"""
    log_key = get_user_chat_log_path(username)
    chat_history = load_chat_history(username)
    
    # 새 메시지 추가
    new_message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    chat_history.append(new_message)
    
    # S3에 저장
    try:
        chatbot_s3_client.put_object(
            Bucket=CHATBOT_BUCKET,
            Key=log_key,
            Body=json.dumps(chat_history, ensure_ascii=False, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        logger.info(f"Chat message saved successfully: {log_key}")
        return True
    except Exception as e:
        logger.error(f"Error saving chat message: {e}")
        return False

# get_chat_context function has been removed as it was redundant with load_chat_history

# ---------------------------
# 4. Pydantic 모델
# ---------------------------

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    action: str
    is_filtered: bool = False
    is_test: bool = False
    emotion: str = "기쁨"  # 기본값으로 '기쁨' 설정

# ---------------------------
# 5. 챗봇 클래스
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
        self.emotion_keywords = {}

    def load_emotion_keywords(self):
        try:
            # S3에서 감정 키워드 파일 다운로드
            emotion_file_path = os.path.join(BASE_DIR, 'emotion_data', 'emotion_keywords.json')
            chatbot_s3_client.download_file(
                Bucket=CHATBOT_BUCKET,
                Key=f"{CHATBOT_PREFIX}emotion_data/emotion_keywords.json",
                Filename=emotion_file_path
            )
            
            # 파일 읽기
            with open(emotion_file_path, 'r', encoding='utf-8') as f:
                self.emotion_keywords = json.load(f)
            
            logger.info("감정 키워드 로드 성공")
            return self.emotion_keywords
        except Exception as e:
            logger.error(f"감정 키워드 로드 실패: {str(e)}")
            return {}

# ---------------------------

@app.get("/")
async def root():
    """루트 경로를 /login으로 리다이렉트합니다."""
    return RedirectResponse(url="/login")

@app.get("/login")
async def login_page(request: Request):
    """로그인 페이지를 반환합니다."""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/chat")
async def chat_page(request: Request):
    """채팅 페이지를 반환합니다."""
    # 로그인되지 않은 사용자는 로그인 페이지로 리다이렉트
    if not request.cookies.get("user_id"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("index.html", {"request": request})

# 프론트엔드와의 호환성을 위해 /chat/ 엔드포인트도 추가
@app.post("/api/chat")
async def handle_chat_message(chat_request: ChatRequest, request: Request):
    """채팅 메시지를 처리합니다."""
    try:
        # 로그인 확인
        user_id = request.cookies.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
        
        # 사용자 정보 확인
        user = member_manager.get_user(user_id)
        if not user:
            response = JSONResponse(
                content={"error": "사용자 정보를 찾을 수 없습니다."},
                status_code=404
            )
            response.delete_cookie(key="user_id")
            return response
        
        # 실제 채팅 처리 로직
        chat_history = []
        
        # 챗봇 응답 생성
        response = await chatbot.generate_response(
            chat_request.message,
            user_id,
            chat_history
        )
        
        # 채팅 기록 저장
        save_chat_message(user_id, "user", chat_request.message)
        save_chat_message(user_id, "assistant", response)
        
        return {
            "response": response,
            "emotion": "중립"
        }
        
    except Exception as e:
        logger.error(f"Error in handle_chat_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", include_in_schema=False)
async def handle_chat(chat_request: ChatRequest, request: Request):
    """채팅 메시지를 처리합니다."""
    try:
        response = await handle_chat_message(chat_request, request)
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error in handle_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error in handle_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_page(request: Request):
    """테스트 페이지를 반환합니다."""
    # 인증 없이도 테스트 페이지 접근 가능
    return templates.TemplateResponse("test.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """회원가입 페이지를 반환합니다."""
    # 이미 로그인한 사용자는 채팅 페이지로 리다이렉트
    user_id = request.cookies.get("user_id")
    if user_id:
        user = member_manager.get_user(user_id)
        if user:
            return RedirectResponse(url="/chat")
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/api/check-auth")
async def check_auth(request: Request):
    """인증 상태 확인 엔드포인트"""
    try:
        user_id = request.cookies.get("user_id")
        if not user_id:
            response = JSONResponse(
                status_code=401,
                content={"detail": "인증되지 않은 사용자입니다."}
            )
            response.headers["Access-Control-Allow-Origin"] = request.headers.get("origin", "*")
            response.headers["Access-Control-Allow-Credentials"] = "true"
            return response
        
        # 사용자 정보 조회
        user = member_manager.get_user(user_id)
        if not user:
            # 사용자 정보가 없으면 쿠키 삭제
            response = JSONResponse(
                status_code=404,
                content={"detail": "사용자 정보를 찾을 수 없습니다."}
            )
            response.delete_cookie(
                key="user_id",
                path="/",
                domain=None,
                secure=False,
                httponly=True
            )
            response.headers["Access-Control-Allow-Origin"] = request.headers.get("origin", "*")
            response.headers["Access-Control-Allow-Credentials"] = "true"
            return response
        
        # 인증 성공
        response_data = {"username": user_id, "name": user.get("name", "")}
        response = JSONResponse(content=response_data)
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("origin", "*")
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response
        
    except Exception as e:
        logger.error(f"인증 확인 중 오류: {str(e)}")
        response = JSONResponse(
            status_code=500,
            content={"detail": "서버 오류가 발생했습니다."}
        )
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("origin", "*")
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

@app.post("/api/login")
async def api_login(login_data: LoginRequest, request: Request):
    """로그인 API 엔드포인트"""
    try:
        # 로그인 처리 로직
        success, message = member_manager.authenticate_user(
            login_data.username,
            login_data.password
        )
        
        if success:
            # 응답 생성
            response_data = {
                "success": True,
                "redirect_url": "/chat",
                "username": login_data.username
            }
            
            response = JSONResponse(
                content=response_data,
                status_code=200
            )
            
            # CORS 헤더 설정
            origin = request.headers.get("origin")
            if origin:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
            
            # 쿠키 설정
            response.set_cookie(
                key="user_id",
                value=login_data.username,
                httponly=True,
                samesite="lax",
                secure=False,  # 개발 환경에서는 False, 프로덕션에서는 True로 설정
                max_age=60 * 60 * 24 * 7,  # 7일
                path="/"
            )
            
            logger.info(f"로그인 성공: {login_data.username}")
            return response
            
        else:
            logger.warning(f"로그인 실패: {login_data.username} - {message}")
            response = JSONResponse(
                status_code=401,
                content={"detail": message}
            )
            origin = request.headers.get("origin")
            if origin:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
            return response
            
    except HTTPException as e:
        logger.error(f"HTTP 에러 발생: {str(e)}")
        response = JSONResponse(
            status_code=e.status_code,
            content={"detail": str(e.detail)}
        )
        origin = request.headers.get("origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        return response
        
    except Exception as e:
        logger.error(f"로그인 처리 중 오류 발생: {str(e)}", exc_info=True)
        response = JSONResponse(
            status_code=500,
            content={"detail": "서버에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요."}
        )
        origin = request.headers.get("origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

@app.get("/test/login")
async def test_login():
    """테스트 페이지로 리다이렉트"""
    return RedirectResponse(url="/test")

@app.get("/api/test/scenarios")
async def get_test_scenarios():
    """S3 버킷에서 테스트 시나리오 목록을 가져옵니다."""
    try:
        logger.info("Fetching test scenarios from S3...")
        
        # 테스트 시나리오는 test 버킷에서 가져옵니다.
        try:
            s3_client = S3Client(bucket_type='test')
            logger.info(f"S3 client initialized with bucket: {s3_client.bucket}, prefix: {s3_client.prefix}")
            
            scenarios = s3_client.list_scenarios()
            logger.info(f"Retrieved {len(scenarios)} scenarios from S3")
            
            # 시나리오가 없는 경우 빈 배열 반환
            if not scenarios:
                logger.info("No test scenarios found in S3 bucket")
                return {"scenarios": []}
            
            # 시나리오 키에서 prefix 제거하여 올바른 키로 반환
            return {"scenarios": [
                {
                    'key': scenario['key'].replace(s3_client.prefix, '', 1),
                    'name': scenario['name']
                }
                for scenario in scenarios
            ]}
            
        except Exception as e:
            logger.error(f"S3 operation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"S3 작업 중 오류가 발생했습니다: {str(e)}"
            )
        
    except HTTPException as he:
        logger.error(f"HTTP error in get_test_scenarios: {str(he.detail)}")
        error_response = JSONResponse(
            status_code=he.status_code,
            content={"detail": str(he.detail)}
        )
        error_response.headers["Access-Control-Allow-Origin"] = "*"
        error_response.headers["Access-Control-Allow-Credentials"] = "true"
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in get_test_scenarios: {str(e)}", exc_info=True)
        error_response = JSONResponse(
            status_code=500,
            content={"detail": f"테스트 시나리오 목록을 불러오는 중 예상치 못한 오류가 발생했습니다: {str(e)}"}
        )
        error_response.headers["Access-Control-Allow-Origin"] = "*"
        error_response.headers["Access-Control-Allow-Credentials"] = "true"
        raise HTTPException(
            status_code=500, 
            detail=f"테스트 시나리오 목록을 불러오는 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        )

@app.get("/api/test/scenario/{scenario_key}")
async def get_scenario(scenario_key: str):
    """특정 테스트 시나리오를 가져옵니다."""
    try:
        # S3 클라이언트 초기화 (테스트 버킷 사용)
        s3_client = S3Client(bucket_type='test')
        
        # 시나리오 키가 완전한 경로가 아닌 경우 경로 추가
        if not scenario_key.startswith(s3_client.prefix):
            scenario_key = f"{s3_client.prefix}{scenario_key}"
        
        # 시나리오 가져오기
        scenario_data = s3_client.get_scenario(scenario_key)
        return scenario_data
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error getting test scenario {scenario_key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"시나리오를 가져오는 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/test/save")
async def save_test_result_endpoint(data: dict):
    """테스트 결과를 S3에 저장합니다."""
    try:
        scenario_key = data.get('scenario_key')
        messages = data.get('messages', [])
        
        if not scenario_key:
            raise HTTPException(status_code=400, detail="시나리오 키가 필요합니다.")
        
        # 테스트 결과 저장 (S3에 저장)
        test_s3_client = S3Client(bucket_type='test')
        
        # 테스트 결과 파일명 생성 (예: results/scenario_key/timestamp.json)
        import time
        from datetime import datetime
        
        timestamp = int(time.time())
        date_str = datetime.utcnow().strftime("%Y%m%d")
        result_key = f"test_results/{date_str}/{scenario_key}_{timestamp}.json"
        
        # 결과 데이터 구성
        result_data = {
            'scenario_key': scenario_key,
            'timestamp': timestamp,
            'created_at': datetime.utcnow().isoformat(),
            'messages': messages
        }
        
        # S3에 저장
        test_s3_client.s3.put_object(
            Bucket=test_s3_client.bucket_name,
            Key=result_key,
            Body=json.dumps(result_data, ensure_ascii=False, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        
        logger.info(f"Test result saved to S3: {result_key}")
        return {"status": "success", "s3_key": result_key}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving test result to S3: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"테스트 결과를 저장하는 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/api/test/save_chat_logs")
async def save_chat_logs(data: dict):
    """테스트 채팅 로그를 S3에 저장합니다."""
    try:
        scenario_key = data.get('scenario_key')
        messages = data.get('messages', [])
        
        if not scenario_key:
            raise HTTPException(status_code=400, detail="시나리오 키가 필요합니다.")
        
        # S3 클라이언트 초기화 (테스트 버킷 사용)
        test_s3_client = S3Client(bucket_type='test')
        
        # 파일명 생성 (S3 키)
        base_name = os.path.basename(scenario_key).replace('.json', '')
        date_str = datetime.utcnow().strftime("%Y%m%d")
        log_key = f"chat_logs/{date_str}/{base_name}_chat_logs.json"
        
        # 한국 시간대 설정
        kst = timezone(timedelta(hours=9))
        
        # 채팅 로그 데이터 준비 (사용자 메시지만 필터링)
        chat_logs = []
        for msg in messages:
            if msg.get('role') == 'user':
                chat_logs.append({
                    'timestamp': datetime.now(kst).isoformat(),  # 한국 시간대 사용
                    'message': msg.get('content', ''),
                    'emotion': msg.get('emotion', '중립')
                })
        
        # 기존 데이터 로드 (있는 경우)
        existing_data = []
        try:
            existing_obj = test_s3_client.s3.get_object(
                Bucket=test_s3_client.bucket_name,
                Key=log_key
            )
            existing_data = json.loads(existing_obj['Body'].read().decode('utf-8'))
        except botocore.exceptions.ClientError as e:
            # 파일이 존재하지 않는 경우 빈 리스트 사용
            pass
        except Exception as e:
            logger.warning(f"기존 채팅 로그를 로드하는 중 오류 발생: {e}")
        
        # 기존 데이터에 새로운 로그 추가
        existing_data.extend(chat_logs)
        
        # S3에 저장
        test_s3_client.s3.put_object(
            Bucket=test_s3_client.bucket_name,
            Key=log_key,
            Body=json.dumps(existing_data, ensure_ascii=False, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        
        logger.info(f"Chat logs saved to S3: {log_key}")
        return {"status": "success", "s3_key": log_key}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving chat logs to S3: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"채팅 로그를 저장하는 중 오류가 발생했습니다: {str(e)}"
        )

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

# 감정별 응답 딕셔너리
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

@app.get("/api/chat-data")
async def get_chat_data():
    # 필요시 실제 데이터로 교체
    return JSONResponse({"categories": ["일상", "업무", "학습", "여행", "음식"]})

@app.post("/api/chat")
async def chat(chat_request: ChatRequest, request: Request):
    """테스트 모드인 경우 감정별 응답 반환"""
    if chat_request.is_test:
        import random
        
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
    
    # 감정 키워드 로드
    try:
        emotion_keywords = chatbot.load_emotion_keywords()
    except Exception as e:
        logger.error(f"감정 키워드 로드 실패: {str(e)}")
        emotion_keywords = {}
    
    # 감정 점수 계산
    try:
        emotion_score = chatbot.calculate_emotion_score(user_message, emotion_keywords)
    except Exception as e:
        logger.error(f"감정 점수 계산 실패: {str(e)}")
        emotion_score = {}
    
    # 챗봇 응답 생성
    response = chatbot.generate_response(
        user_input=user_message,
        username=username,
        chat_history=chat_context
    )
    
    save_chat_message(username, "assistant", response)
    return {"response": response, "status": "success", "emotion_score": emotion_score}
