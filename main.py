import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import os
import json
import spacy
import shutil
import uvicorn
import contextualSpellCheck
from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_MAP_PATH = os.path.join(BASE_DIR, "mapping", "sign_to_prediction_index_map.json")
MODEL_PATH = os.path.join(BASE_DIR, "LSTM_weights", "best_asl_model_avg_66_modified.pt")
LLM_MODEL_PATH = os.path.join(BASE_DIR, "model_qwen_files")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



nlp = spacy.load("en_core_web_sm")
if "contextualSpellCheck" not in nlp.pipe_names:
    try:
        contextualSpellCheck.add_to_pipe(nlp)
    except:
        print("Warning: Could not load ContextualSpellCheck, using basic Spacy.")


with open(JSON_MAP_PATH, 'r') as f:
    sign_map = json.load(f)
label_names = [None] * len(sign_map)
for name, idx in sign_map.items():
    label_names[idx] = name


tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, local_files_only=True)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, device_map="auto", torch_dtype="auto", local_files_only=True)


class DualStreamLSTM(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.face_fc = nn.Linear(1404, 256)
        self.face_lstm = nn.LSTM(256, 256, batch_first=True, num_layers=2, dropout=0.3)
        self.body_fc = nn.Linear(225, 256)
        self.body_lstm = nn.LSTM(256, 256, batch_first=True, num_layers=2, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )
    def forward(self, face, body):
        f = torch.relu(self.face_fc(face))
        _, (h_f, _) = self.face_lstm(f)
        b = torch.relu(self.body_fc(body))
        _, (h_b, _) = self.body_lstm(b)
        combined = (h_f[-1] + h_b[-1]) / 2
        return self.classifier(combined)


model_asl = DualStreamLSTM(len(label_names)).to(device)
model_asl.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model_asl.eval()


def refine_with_spacy(text):
    doc = nlp(text)
    
    sentence = doc._.outcome_spellCheck if doc._.performed_spellCheck else doc.text
    sentence = sentence.strip()
    if len(sentence) > 0:
        
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith('.'): 
            sentence += '.'
    return sentence

def text_generation(sign_words):
    prompt = f"The user signed these words: '{sign_words}'. Write a natural English sentence using them."
    messages = [{"role": "system", "content": "You are a sign language interpreter. Output ONLY the final sentence."},
                {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generate_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    ids = llm_model.generate(inputs.input_ids, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

def predict_segment(segment_lms):
    target_frames = 30
    n_input_frames = len(segment_lms)
    if n_input_frames >= target_frames:
        indices = np.linspace(0, n_input_frames - 1, target_frames).astype(int)
    else:
        indices = np.pad(np.arange(n_input_frames), (0, target_frames - n_input_frames), 'edge')
    
    data = segment_lms[indices]
    face = np.nan_to_num(data[:, 0:468, :].reshape(target_frames, -1), nan=0.0)
    body = np.nan_to_num(data[:, 468:, :].reshape(target_frames, -1), nan=0.0)
    
    face = (face - face.mean()) / (face.std() + 1e-6)
    body = (body - body.mean()) / (body.std() + 1e-6)
    
    face_t = torch.tensor(face).float().unsqueeze(0).to(device)
    body_t = torch.tensor(body).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model_asl(face_t, body_t)
        idx = out.argmax(1).item()
        conf = torch.softmax(out, 1).max().item()
    return label_names[idx], conf



app = FastAPI()

@app.post("/translate")
async def ish_translator(video: UploadFile = File(...)):
    temp_path = os.path.join(BASE_DIR, f"temp_{video.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    try:
        
        cap = cv2.VideoCapture(temp_path)
        frames_lms = []
        mp_holistic = mp.solutions.holistic
        
        with mp_holistic.Holistic() as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                lms = np.zeros((543, 3))
                if results.face_landmarks:
                    for i, lm in enumerate(results.face_landmarks.landmark): lms[i] = [lm.x, lm.y, lm.z]
                if results.left_hand_landmarks:
                    for i, lm in enumerate(results.left_hand_landmarks.landmark): lms[468+i] = [lm.x, lm.y, lm.z]
                if results.pose_landmarks:
                    for i, lm in enumerate(results.pose_landmarks.landmark): 
                        if i < 33: lms[489+i] = [lm.x, lm.y, lm.z]
                if results.right_hand_landmarks:
                    for i, lm in enumerate(results.right_hand_landmarks.landmark): lms[522+i] = [lm.x, lm.y, lm.z]
                frames_lms.append(lms)
        cap.release()

        if not frames_lms:
            return {"error": "No landmarks detected"}

        label, conf = predict_segment(np.array(frames_lms))
        
        if conf > 0.40:
            generated = text_generation(label)
            final_sentence = refine_with_spacy(generated)
            return {
                "prediction": label,
                "confidence": round(conf, 4),
                "final_sentence": final_sentence
            }
        else:
            return {"prediction": label, "confidence": round(conf, 4), "note": "Low confidence"}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)