# 감정 인식 챗봇

이 프로젝트는 사용자의 대화를 분석하여 감정을 인식하고, 그에 맞는 공감적인 응답을 생성하는 챗봇입니다.

## 기능

- 사용자 메시지의 감정 분석 (기쁨, 분노, 불안, 슬픔, 중립, 혐오, 당황)
- 감정에 맞는 공감적 응답 생성
- 대화 내내 감정 추적 및 요약 제공

## 설치 방법

1. 저장소를 클론합니다:
   ```bash
   git clone [your-repository-url]
   cd chatbot
   ```

2. 가상 환경 생성 및 활성화 (선택 사항):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 또는
   .\venv\Scripts\activate  # Windows
   ```

3. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 사용 방법

1. 챗봇 실행:
   ```bash
   python chatbot.py
   ```

2. 대화 시작:
   - 챗봇과 자유롭게 대화하세요.
   - 챗봇이 자동으로 감정을 분석하고 그에 맞는 응답을 생성합니다.
   - 대화를 종료하려면 '종료'를 입력하세요.

3. 대화 종료 시:
   - 대화 중 감정 분포가 요약되어 표시됩니다.
   - 가장 지배적인 감정이 무엇인지 확인할 수 있습니다.

## 모델 정보

- **감정 분석 모델**: monologg/koelectra-base-v3-discriminator
- **챗봇 모델**: skt/kogpt2-base-v2

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.
