import streamlit as st
import requests
import json
import time

# Cấu hình trang
st.set_page_config(
    page_title="VNPT RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Khởi tạo session state cho lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Cấu hình")
    api_url = st.text_input("FastAPI URL", value="http://localhost:8000")
    st.divider()
    
    st.subheader("📚 Quản lý tri thức")
    if st.button("Nạp lại dữ liệu (Ingest)", use_container_width=True):
        with st.spinner("Đang nạp dữ liệu vào Vector Database..."):
            try:
                response = requests.post(f"{api_url}/ingest", json={})
                if response.status_code == 200:
                    st.success(response.json().get("message"))
                else:
                    st.error(f"Lỗi: {response.text}")
            except Exception as e:
                st.error(f"Không thể kết nối API: {e}")

    st.divider()
    st.caption("Phát triển bởi RAG Team - Học qua hành")

# --- Main UI ---
st.title("🤖 Chatbot Hỗ trợ Tri thức")
st.markdown("Hệ thống RAG sử dụng **Qwen 2.5 3B** và **Hybrid Search** (FAISS + BM25).")

# Hiển thị lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📌 Nguồn trích dẫn"):
                for src in set(message["sources"]):
                    st.caption(f"- {src}")

# Ô nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi API để lấy câu trả lời
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Đang lục tìm tài liệu..."):
                response = requests.post(
                    f"{api_url}/ask", 
                    json={"question": prompt},
                    timeout=300
                )
            
            if response.status_code == 200:
                result = response.json()
                full_response = result["answer"]
                sources = result["sources"]
                
                # Giả lập hiệu ứng streaming (vì Ollama trả về 1 cục)
                for chunk in full_response.split():
                    message_placeholder.markdown(full_response[:full_response.find(chunk) + len(chunk)] + "▌")
                    time.sleep(0.05)
                
                message_placeholder.markdown(full_response)
                
                if sources:
                    with st.expander("📌 Nguồn trích dẫn"):
                        for src in set(sources):
                            st.caption(f"- {src}")
                
                # Lưu vào lịch sử
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })
            else:
                st.error(f"Lỗi API: {response.status_code}")
        except Exception as e:
            st.error(f"Lỗi kết nối: {e}")
