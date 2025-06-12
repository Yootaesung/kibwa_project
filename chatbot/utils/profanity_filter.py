# 욕설 필터링 모듈
import re

class ProfanityFilter:
    def __init__(self):
        self.load_profanities()

    def load_profanities(self):
        import json
        import os
        
        # JSON 파일 경로
        json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'profanity_data', 'korean_profanities.json')
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.profanities = data['profanities']
        except Exception as e:
            print(f"Error loading profanities: {e}")
            self.profanities = {'basic': [], 'slang': []}

    def create_pattern(self, word, variants):
        # 단어와 변형들로 정규표현식 패턴 생성
        patterns = [f'\b{word}\b']
        for variant in variants:
            patterns.append(f'\b{variant}\b')
        return '|'.join(patterns)

    def filter_text(self, text):
        filtered_text = text
        
        # 기본 욕설 필터링
        for profanity in self.profanities['basic']:
            pattern = self.create_pattern(profanity['word'], profanity['variants'])
            filtered_text = re.sub(pattern, profanity['replacement'], filtered_text, flags=re.IGNORECASE)
        
        # 슬랭 욕설 필터링
        for profanity in self.profanities['slang']:
            pattern = self.create_pattern(profanity['word'], profanity['variants'])
            filtered_text = re.sub(pattern, profanity['replacement'], filtered_text, flags=re.IGNORECASE)
        
        return filtered_text

    def has_profanity(self, text):
        """텍스트에 욕설이 포함되어 있는지 확인"""
        if not text:
            return False
            
        for profanity in self.profanities['basic']:
            pattern = self.create_pattern(profanity['word'], profanity['variants'])
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        for profanity in self.profanities['slang']:
            pattern = self.create_pattern(profanity['word'], profanity['variants'])
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        return False

# 전역 필터 인스턴스 생성
profanity_filter = ProfanityFilter()
