import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from collections import Counter
import string
from typing import List, Dict, Tuple
import re
import logging

# Cấu hình logging
logger = logging.getLogger(__name__)

# Download required NLTK data (chỉ download những thứ cơ bản nhất)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK 'punkt'...")
    nltk.download('punkt', quiet=True)

# DANH SÁCH STOPWORDS TIẾNG VIỆT (Rút gọn cho hiệu suất)
# NLTK thường không có sẵn tiếng Việt tốt, nên ta tự định nghĩa để đảm bảo hoạt động.
VIETNAMESE_STOPWORDS = {
    'là', 'và', 'của', 'thì', 'mà', 'có', 'những', 'trong', 'một', 'các', 'cho', 'với', 
    'được', 'khi', 'này', 'đó', 'như', 'để', 'cũng', 'của', 'người', 'ra', 'vào', 'lại', 
    'đến', 'bằng', 'về', 'nhưng', 'năm', 'theo', 'lên', 'xuống', 'nó', 'gì', 'ai', 'tôi',
    'bạn', 'cô', 'anh', 'chúng', 'tại', 'bởi', 'vì', 'dù', 'rằng', 'sau', 'trước', 'trên', 
    'dưới', 'từ', 'hơn', 'rất', 'sẽ', 'đã', 'đang', 'chỉ', 'thể', 'việc', 'loại', 'cách'
}

class TextAnalyzer:
    """Phân tích nội dung văn bản chuyên dụng cho hệ thống UPT (Hỗ trợ Tiếng Việt/Anh)"""
    
    def __init__(self, language: str = 'vietnamese'):
        self.language = language
        self.punctuation = set(string.punctuation)
        
        # Thiết lập stopwords
        self.stop_words = set()
        if language == 'vietnamese':
            self.stop_words = VIETNAMESE_STOPWORDS
        else:
            try:
                from nltk.corpus import stopwords
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('english'))
            except Exception:
                self.stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an'} # Fallback tối thiểu

    def preprocess_text(self, text: str) -> List[str]:
        """Tiền xử lý văn bản: Làm sạch, tách từ và lọc rác"""
        if not text:
            return []
            
        # 1. Chuyển về chữ thường
        text = text.lower()
        
        # 2. Xóa ký tự đặc biệt (giữ lại chữ cái và số có ý nghĩa, loại bỏ dấu câu)
        # Đối với tiếng Việt, ta chấp nhận các ký tự unicode
        text = re.sub(r'[^\w\s]', ' ', text) 
        
        # 3. Xóa số thuần túy (để tránh nhiễu keyword như '1', '2024')
        text = re.sub(r'\b\d+\b', '', text)
        
        # 4. Tách từ (Tokenization)
        try:
            words = word_tokenize(text)
        except Exception:
            words = text.split() # Fallback nếu NLTK lỗi
            
        # 5. Lọc Stopwords và từ quá ngắn
        clean_words = []
        for word in words:
            word = word.strip()
            if (len(word) > 2 and 
                word not in self.stop_words and 
                word not in self.punctuation):
                clean_words.append(word)
                
        return clean_words
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """Trích xuất từ khóa chính dựa trên tần suất"""
        words = self.preprocess_text(text)
        if not words:
            return []
            
        word_freq = FreqDist(words)
        return word_freq.most_common(top_n)
    
    def summarize_text(self, text: str, num_sentences: int = 3) -> str:
        """Tóm tắt văn bản dựa trên trọng số từ khóa"""
        if not text:
            return ""
            
        # Tách câu
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = text.split('.') # Fallback đơn giản
            
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Tính tần suất từ toàn văn bản
        words = self.preprocess_text(text)
        word_freq = FreqDist(words)
        if not word_freq:
            return ' '.join(sentences[:num_sentences])
            
        max_freq = word_freq.most_common(1)[0][1]
        
        # Chuẩn hóa tần suất
        for w in word_freq:
            word_freq[w] = (word_freq[w] / max_freq)
            
        # Tính điểm cho từng câu
        sent_scores = {}
        for sent in sentences:
            for word in self.preprocess_text(sent):
                if word in word_freq:
                    if sent not in sent_scores:
                        sent_scores[sent] = word_freq[word]
                    else:
                        sent_scores[sent] += word_freq[word]
        
        # Chọn ra các câu điểm cao nhất
        import heapq
        summary_sentences = heapq.nlargest(num_sentences, sent_scores, key=sent_scores.get)
        
        return ' '.join(summary_sentences)
    
    def classify_content(self, text: str) -> str:
        """
        Phân loại nội dung theo hệ quy chiếu UPT.
        Hỗ trợ cả từ khóa Tiếng Anh và Tiếng Việt.
        """
        categories = {
            'Lý thuyết UPT': [
                'upt', 'pulse', 'unified', 'theory', 'resonance', 'coherence', 'dao động', 
                'cộng hưởng', 'đồng bộ', 'tần số', 'xung', 'lý thuyết', 'hệ thống', 'cấu trúc'
            ],
            'Vật lý Lượng tử': [
                'quantum', 'physics', 'nuclear', 'atom', 'plasma', 'energy', 'reactor', 
                'lượng tử', 'vật lý', 'hạt nhân', 'nguyên tử', 'năng lượng', 'lò phản ứng'
            ],
            'Khoa học Ý thức': [
                'consciousness', 'mind', 'brain', 'cognitive', 'awareness', 'psychology', 
                'ý thức', 'tâm trí', 'nhận thức', 'não bộ', 'tâm lý', 'cảm xúc'
            ],
            'AI & Công nghệ': [
                'ai', 'intelligence', 'artificial', 'model', 'data', 'code', 'python', 'algorithm',
                'trí tuệ', 'nhân tạo', 'mô hình', 'dữ liệu', 'thuật toán', 'lập trình'
            ],
            'Tài chính & Xã hội': [
                'money', 'finance', 'social', 'human', 'world', 'life',
                'tiền', 'tài chính', 'xã hội', 'con người', 'cuộc sống', 'thế giới'
            ]
        }
        
        words = self.preprocess_text(text)
        word_freq = FreqDist(words)
        
        scores = {cat: 0 for cat in categories}
        
        for category, keywords in categories.items():
            for keyword in keywords:
                # Tìm kiếm keyword (xử lý một phần cho tiếng Việt đơn giản)
                if keyword in word_freq:
                    scores[category] += word_freq[keyword]
        
        # Kiểm tra nếu không có từ khóa nào khớp
        if max(scores.values()) == 0:
            return 'Chưa phân loại'
            
        # Trả về category có điểm cao nhất
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def analyze(self, text: str) -> Dict:
        """Phân tích toàn bộ văn bản và trả về kết quả tổng hợp"""
        try:
            return {
                'summary': self.summarize_text(text),
                'keywords': self.extract_keywords(text),
                'category': self.classify_content(text)
            }
        except Exception as e:
            logger.error(f"Lỗi trong quá trình analyze: {e}")
            return {
                'summary': "Lỗi phân tích",
                'keywords': [],
                'category': "Lỗi"
            }