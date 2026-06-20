Ishara AI Engine
Project Overview
This folder contains the core intelligence of the Ishara project. It includes the deep learning architectures for sign language recognition and the Natural Language Generation (NLG) system.

Large Files Access (Google Drive)
Due to size limits, the heavy model files must be downloaded from Google Drive and placed in their respective folders as shown below:

Download Models & Weights from Google Drive

Required Folders from Drive:
LSTM_weights/ — Contains the trained LSTM weights (.pt file).
model_qwen_files/ — Contains the Qwen NLP model files.
Technical Stack
Deep Learning Framework: PyTorch (Used for the Dual-Stream LSTM architecture).
Computer Vision: MediaPipe Holistic (For 3D landmark extraction: Face, Hands, and Pose).
NLP & Language Modeling: * Qwen LLM: For natural sentence generation.
SpaCy & contextualSpellCheck: For grammatical refinement[cite: 2].
Backend API: FastAPI & Uvicorn[cite: 2].
Data Processing: NumPy & Pandas[cite: 2]
Setup & Execution Guide
1. Linking Models
After downloading the files from Drive, your local directory must look like this to match the code paths:

/MachineLearning_Models
├── main.py
├── requirements.txt
├── models/
│   └── model_utils.py                    <-- [Training logic & Model helpers]
├── mapping/
│   └── sign_to_prediction_index_map.json
├── LSTM_weights/                          <-- [Place .pt file here]
│   └── best_asl_model_avg_66_modified.pt  <-- [Download from Drive]
└── model_qwen_files/                      <-- [Place Qwen files here]
    ├── config.json                        <-- [Download from Drive]
    ├── model.safetensors                  <-- [Download from Drive]
    ├── tokenizer.json                     <-- [Download from Drive]
    └── generation_config.json             <-- [Download from Drive]
2.Installation

pip install -r requirements.txt
python -m spacy download en_core_web_sm
Local Inference
uvicorn main:app --host 0.0.0.0 --port 8000
API Endpoints :

POST /translate: Receives video/landmarks and returns the predicted sentence.

GET /docs: Interactive Swagger documentation.
