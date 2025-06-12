# Kibwa Chatbot

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

한국어 감정 인식 챗봇으로, 사용자의 감정을 이해하고 공감하는 대화를 나눌 수 있는 인공지능 챗봇입니다. OpenAI의 GPT 모델을 기반으로 하여 자연스러운 대화가 가능합니다.

## ✨ 주요 기능

- **실시간 감정 분석**: 사용자의 메시지에서 감정을 실시간으로 분석
- **맥락 이해**: 대화의 흐름을 이해하여 일관된 응답 제공
- **대화 기록**: 대화 내용을 자동으로 저장하여 이력 관리
- **맞춤형 응답**: 사용자의 감정 상태에 따라 다양한 응답 제공
- **쉬운 통합**: RESTful API를 통한 손쉬운 통합 지원

## 🚀 시작하기

### 사전 요구사항

- Python 3.8 이상
- OpenAI API 키
- pip (Python 패키지 관리자)

### 설치 방법

1. 저장소를 클론합니다:
   ```bash
   git clone https://github.com/yourusername/kibwa-chatbot.git
   cd kibwa-chatbot
   ```

2. 가상 환경을 생성하고 활성화합니다:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 또는
   .\venv\Scripts\activate  # Windows
   ```

3. 의존성을 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

4. 환경 변수를 설정합니다 (`.env` 파일 생성):
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   SECRET_KEY=your_secret_key_here
   DEBUG=True
   ```

## 🛠️ 사용 방법

### 기본 사용 예제

```python
from chatbot import EmotionAwareChatbot

# 챗봇 인스턴스 생성
chatbot = EmotionAwareChatbot()

# 대화 시작
response = chatbot.generate_response("오늘 기분이 너무 좋아!")
print(f"챗봇: {response}")

# 대화 기록 저장
chatbot.save_conversation("conversation_001.json")
```

### FastAPI 서버 실행

```bash
uvicorn chatbot.app:app --reload
```

API 문서는 `http://localhost:8000/docs`에서 확인할 수 있습니다.

## 📂 프로젝트 구조

```
kibwa-chatbot/
├── chatbot/                    # 주요 패키지
│   ├── __init__.py
│   ├── app.py                  # FastAPI 애플리케이션
│   ├── chatbot.py              # 챗봇 핵심 로직
│   ├── member.py               # 사용자 관리
│   └── config/                 # 설정 파일들
│       ├── __init__.py
│       ├── settings.py         # 애플리케이션 설정
│       └── logger.py           # 로깅 설정
├── tests/                      # 테스트 코드
├── static/                     # 정적 파일 (CSS, JS, 이미지 등)
├── templates/                  # HTML 템플릿
├── .env.example                # 환경 변수 예시
├── requirements.txt            # Python 의존성
├── setup.py                    # 패키지 설정
└── README.md                   # 이 파일
```

## 🤝 기여하기

기여를 환영합니다! 버그 리포트, 기능 제안, 풀 리퀘스트 등 모든 형태의 기여를 환영합니다.

1. 포크하고 저장소를 클론합니다.
2. 새로운 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`).
3. 변경 사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`).
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`).
5. 풀 리퀘스트를 엽니다.

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 연락처

질문이나 제안이 있으시면 다음으로 연락주세요:
- 이메일: contact@kibwa.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

<div align="center">
  <sub>만든이 ❤️ Kibwa Team | 2023</sub>
</div>