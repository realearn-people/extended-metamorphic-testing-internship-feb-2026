import pandas as pd
import requests
from tqdm import tqdm
from collections import Counter
import os

# --- 1. ตั้งค่าพื้นฐาน ---
MODEL_NAME = "qwq:32b" 
OLLAMA_URL = "http://localhost:11434/api/chat"
INPUT_FILE = 'sentiment_temporal.csv'
OUTPUT_FILE = 'qwq_result_sentiment_temporal_tem1.csv'

def ask_ollama(sentence):

    system_content = (
        """""You are an assistant that classifies the sentiment of the message into positive, negative, and neutral. Given below is an example of the sentiment analysis task.\n\n 
        Sentence: I had a bad experience\n"
        Sentiment: Negative"""""
    )
    # ใช้ User Prompt ตามที่คุณกำหนดเป๊ะๆ
    user_content = (
        f"What is the sentiment of the following sentence?"
        f"Limit your answer to only one of these options: Positive, Negative, or Neutral.\n"
        f"{sentence}"
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        "stream": False,
        "options": {
            "temperature": 1,
            "top_p": 0.07,
            "num_predict": 512
        }
    }
    
    try:
    
        # --- ส่วนที่ควรปรับในฟังก์ชัน ask_ollama ---
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        raw_ans = response.json()['message']['content'].strip().lower()
        
        # แยกคำออกมาเป็น list และดูเฉพาะส่วนท้ายๆ หรือเช็คแบบ Case-by-case
        # หรือใช้การเช็คที่เข้มงวดขึ้นเพื่อไม่ให้โดนตัว "Thought" หลอก
        
        # วิธีที่ 1: เช็คจากคำสุดท้ายที่โมเดลตอบ (มักจะเป็นคำตอบสรุป)
        words = raw_ans.split()
        if not words: return "Invalid"
        
        # วนลูปเช็คจากคำหลังสุดมาหน้าสุด
        for word in reversed(words):
            clean_word = word.strip('.,!?:;"()')
            if clean_word == 'positive': return 'Positive'
            if clean_word == 'negative': return 'Negative'
            if clean_word == 'neutral': return 'Neutral'
        
        return "Invalid"
    except:
        return "Error"

# --- 2. ฟังก์ชันรันข้อมูลทุกแถว ---
def process_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: ไม่พบไฟล์ {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"กำลังเริ่มประมวลผลทั้งหมด {len(df)} แถว...")

    p1, p2, p3, final = [], [], [], []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        sentence = row['Perturbed']
        
        # รัน 3 รอบทำ Majority Vote
        v1 = ask_ollama(sentence)
        v2 = ask_ollama(sentence)
        v3 = ask_ollama(sentence)
        
        p1.append(v1); p2.append(v2); p3.append(v3)
        
        # สรุปผล
        valid_votes = [v for v in [v1, v2, v3] if v in ["Positive", "Negative", "Neutral"]]
        winner = Counter(valid_votes).most_common(1)[0][0] if valid_votes else v1
        final.append(winner)

    # เก็บผลลัพธ์และตรวจความถูกต้อง
    df['prediction1'], df['prediction2'], df['prediction3'] = p1, p2, p3
    df['Prediction'] = final
    df['Is_Correct'] = df['Prediction'].str.lower() == df['Expected_answer'].str.lower()
    
    # บันทึกไฟล์
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    accuracy = (df['Is_Correct'].sum() / len(df)) * 100
    print(f"\nประมวลผลสำเร็จ! Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    process_data()