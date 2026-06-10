import requests

from rag_chatbot.config import get_settings
from rag_chatbot.protocols import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model: str, endpoint: str = "http://localhost:11434/api/chat"):
        self.model = model
        self.endpoint = endpoint

    def generate(self, question: str, context: str) -> str:
        settings = get_settings()
        system_prompt = (
            "Bạn là trợ lý AI chuyên nghiệp giải đáp thắc mắc dựa trên tài liệu nội bộ bằng tiếng Việt.\n"
            "Nhiệm vụ của bạn là trả lời CÂU HỎI chỉ sử dụng thông tin thực tế từ phần CONTEXT trong tin nhắn của người dùng.\n"
            "CÁC QUY TẮC BẮT BUỘC:\n"
            "1. Chỉ trả lời dựa hoàn toàn vào thông tin được cung cấp trong phần CONTEXT. Tuyệt đối KHÔNG tự bịa đặt câu trả lời, KHÔNG suy diễn và KHÔNG sử dụng kiến thức bên ngoài của bạn.\n"
            "2. Cực kỳ cẩn thận với các từ chỉ mốc thời gian như 'hiện tại', 'ngày nay', 'bây giờ', 'mới nhất'. Nếu tài liệu chỉ đề cập đến thông tin trong quá khứ (ví dụ: 'vào năm 2016', 'thời xưa') mà không khẳng định rõ ràng đó vẫn là thông tin của hiện tại, bạn BẮT BUỘC phải xác định là thông tin không đủ tin cậy và trả lời chính xác cụm từ sau: 'Tôi không tìm thấy thông tin đủ tin cậy trong tài liệu'.\n"
            "3. Nếu phần CONTEXT không chứa thông tin trực tiếp để trả lời cho câu hỏi, bạn BẮT BUỘC phải trả lời chính xác cụm từ sau và không thêm gì khác: 'Tôi không tìm thấy thông tin đủ tin cậy trong tài liệu'.\n"
            "4. Câu trả lời cần ngắn gọn, súc tích, khách quan và đi thẳng vào câu hỏi."
        )
        user_content = f"CONTEXT:\n{context}\n\nCÂU HỎI:\n{question}"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "stream": False,
            "options": {
                "temperature": settings.temperature
            }
        }
        try:
            # Tăng timeout lên 300s vì chạy local có thể chậm
            response = requests.post(self.endpoint, json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"Lỗi khi gọi Ollama Chat API: {e}")
            return "Lỗi: Không thể nhận câu trả lời từ model."


