# extended-metamorphic-testing-internship-feb-2026
This is a repository that contains the experiments jointly conducted with internship students in February 2026. 

## Benjawan

จะเก็บข้อมูลย่อยทุกอย่าง (code, ผลลัพธ์) ไว้ใน folder หลักตามชื่อของโมเดลที่ใช้ในการเทรนข้อมูล ซึ่งจะมีอยู่ทั้งหมด 3 folder ประกอบด้วย :
- Deepseek-v2_16b Model
- Gemma3_270m Model
- QWQ_32b Model
และจะเก็บผลรวมของทุกโมเดลไว้ในไฟล์ :
- ALL_MODELS_sentiment_analysis_result.csv

โดยในแต่ละ folder หลัก ข้างในจะประกอบไปด้วย :
- Deepseek-v2_16b Model
   - Analsis_data #ผลรวมแยกตามแต่ละ dataset (7 dataset)
   - Deepseek_model_result_tem0 #ผลของแต่ละ dataset แบ่งตาม tem ในนี้จะเป็น tem = 0
   - Deepseek_model_result_tem1 #ผลของแต่ละ dataset แบ่งตาม tem ในนี้จะเป็น tem = 1
   - Deepseek-v2_16b_sentiment_alldata_analysis_result.csv # ไฟล์รวมผลของทุก dataset ในโมเดล deepseek
   - deepseek_model.py
 
- Gemma3_270m Model
   - Analsis_data #ผลรวมแยกตามแต่ละ dataset (7 dataset)
   - Gemma_model_result_tem0 #ผลของแต่ละ dataset แบ่งตาม tem ในนี้จะเป็น tem = 0
   - Gemma_model_result_tem1 #ผลของแต่ละ dataset แบ่งตาม tem ในนี้จะเป็น tem = 1
   - Gemma3_270m_sentiment_alldata_analysis_result.csv # ไฟล์รวมผลของทุก dataset ในโมเดล gemma
   - gemma_model.py

- QWQ_32b Model
   - Analsis_data #ผลรวมแยกตามแต่ละ dataset (7 dataset)
   - QWQ_model_result_tem0 #ผลของแต่ละ dataset แบ่งตาม tem ในนี้จะเป็น tem = 0
   - QWQ_model_result_tem1 #ผลของแต่ละ dataset แบ่งตาม tem ในนี้จะเป็น tem = 1
   - QWQ_32b_sentiment_alldata_analysis_result.csv # ไฟล์รวมผลของทุก dataset ในโมเดล qwq
   - qwq_model.py

## Chanidapha
- Dataset: เก็บชุดข้อมูลที่ผ่านการปรับเปลี่ยนในการใช้ทำการทดลอง 7 รูปแบบ
- Result: เก็บผลลัพธ์จากการรันโมเดลแยกตามชื่อและพารามิเตอร์
  - แยกโฟลเดอร์ตามโมเดล
  - ในโฟลเดอร์แต่ละโมเดลแยกย่อยตามค่า Temperature
  - Global_Analysis_Result.csv: ไฟล์สรุปผลภาพรวมของทุกการทดลอง
- SourceCode: โค้ดที่ใช้ในการประมวลผล
  - Execution_script.ipynb: สคริปต์ที่ใช้ประเมินผลการทดลองทั้งหมด
  - สคริปต์รายโมเดลสำหรับการรัน ได้แก่ Deepseek-r1_8b_script.ipynb, Gemma3_1b_script.ipynb และ Qwen2.5_3b_script.ipynb
