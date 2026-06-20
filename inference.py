import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import mediapipe as mp
import os
import json
import spacy
import contextualSpellCheck
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load English model for NLP tasks
nlp = spacy.load("en_core_web_sm")
if "contextualSpellCheck" not in nlp.pipe_names:
    contextualSpellCheck.add_to_pipe(nlp)


#Uses SpaCy to correct spelling and format the sentence correctly
def refine_with_spacy(text):
    doc = nlp(text)
    # Check if a spell-checked version exists, otherwise use original text
    sentence = doc._.outcome_spellCheck if doc._.performed_spellCheck else doc.text
    sentence = sentence.strip()
    if len(sentence) > 0:
    # Capitalize the first letter and ensure the sentence ends with a period
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith('.'): sentence += '.'
    return sentence

#Paths and Device Configuration
JSON_MAP_PATH = r"C:\Users\pc\Downloads\sign_to_prediction_index_map.json"
MODEL_PATH = r"C:\Users\pc\Downloads\best_asl_model_avg_66_modified.pt"
LLM_MODEL_PATH = r"C:\Users\pc\Downloads\archive (1)\model_qwen_files"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading Label Map
# Essential for matching model output indices to actual sign names
with open(JSON_MAP_PATH, 'r') as f:
    sign_map = json.load(f)
# Convert dictionary to an ordered list based on the index
label_names = [None] * len(sign_map)
for name, idx in sign_map.items():
    label_names[idx] = name

#Loading Models (ASL Classifier & LLM) 
# Load Tokenizer and Large Language Model (LLM) for text generation
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, local_files_only=True)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, device_map="auto", torch_dtype="auto", local_files_only=True)

class DualStreamLSTM(nn.Module):
    def __init__(self, n_classes):
        #Model architecture featuring two parallel streams for Face and Body landmarks
        super().__init__()
        # Face stream layers
        self.face_fc = nn.Linear(1404, 256)
        self.face_lstm = nn.LSTM(256, 256, batch_first=True, num_layers=2, dropout=0.3)
        # Body/Hands stream layers
        self.body_fc = nn.Linear(225, 256)
        self.body_lstm = nn.LSTM(256, 256, batch_first=True, num_layers=2, dropout=0.3)
        # Classifier head
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
        # Average the last hidden states of both streams
        combined = (h_f[-1] + h_b[-1]) / 2
        return self.classifier(combined)

# Initialize and load the custom ASL model
model = DualStreamLSTM(len(label_names)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

#Helper Functions for Prediction

def text_generation(sign_words):
    #Prompts the LLM to convert a list of predicted words into a natural sentence
    prompt = f"The user signed these words: '{sign_words}'. Write a natural English sentence using them."
    messages = [{"role": "system", "content": "You are a sign language interpreter. Output ONLY the final sentence."},
                {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generate_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    # Generate response using the LLM
    ids = llm_model.generate(inputs.input_ids, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

def predict_segment(segment_lms):
    
#Processes landmark arrays and generates a sign prediction
    
    target_frames = 30
    n_input_frames = len(segment_lms)
    
    #Resampling and Padding to match training input size (30 frames)
    if n_input_frames >= target_frames:
        indices = np.linspace(0, n_input_frames - 1, target_frames).astype(int)
    else:
        indices = np.pad(np.arange(n_input_frames), (0, target_frames - n_input_frames), 'edge')
    
    data = segment_lms[indices]
    
    #Split landmarks into Face and Body components (Expected shape: 30, 543, 3)
    face = data[:, 0:468, :].reshape(target_frames, -1)
    body = data[:, 468:, :].reshape(target_frames, -1)
    
    #Handle missing values
    face = np.nan_to_num(face, nan=0.0)
    body = np.nan_to_num(body, nan=0.0)
    
    # Normalization (matching training statistics)
    face = (face - face.mean()) / (face.std() + 1e-6)
    body = (body - body.mean()) / (body.std() + 1e-6)
    
    # Convert to Tensors and perform Inference
    face_t = torch.tensor(face).float().unsqueeze(0).to(device)
    body_t = torch.tensor(body).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(face_t, body_t)
        idx = out.argmax(1).item()
        conf = torch.softmax(out, 1).max().item()
        
    return label_names[idx], conf

#Main Video Processing Function
def process_video(video_path):
    
#Captures video, extracts landmarks per frame, and returns predicted text.
    cap = cv2.VideoCapture(video_path)
    frames_lms = []
    mp_holistic = mp.solutions.holistic
    
    print("Extracting landmarks...")
    with mp_holistic.Holistic() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # Convert frame to RGB for Mediapipe
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lms = np.zeros((543, 3))

            # Map landmarks to standard array structure
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
        return "No landmarks detected in the video."

    # For single-word video tests, process all captured frames as one segment
    frames_lms_array = np.array(frames_lms)
    label, conf = predict_segment(frames_lms_array)
    
    print("-" * 30)
    print(f"Detected Word: {label}")
    print(f"Confidence: {conf:.4f}")
    print("-" * 30)


    # Thresholding for reliability
    if conf > 0.40:
        # Generate and refine the final output sentence
        generated = text_generation(label)
        final = refine_with_spacy(generated)
        return final
    else:
        return f"Confidence too low ({conf:.4f}). Prediction: {label}"

# --- Execution ---
video_test_path = r"C:\Users\pc\Downloads\test_signs\no_sign_test.mp4"
result = process_video(video_test_path)
print(f"Final Output: {result}")