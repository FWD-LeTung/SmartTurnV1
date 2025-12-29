import gradio as gr
import numpy as np 
import librosa
from inference import predict_endpoint, predict_endpoint2

def predict(audio_file):
    if audio_file is None:
        return "Vui lòng tải lên một tệp âm thanh."
        
    audio, sr = librosa.load(audio_file)
    
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
        
    result = predict_endpoint(audio)
    return result

def predict2(audio_file):
    if audio_file is None:
        return "Vui lòng tải lên một tệp âm thanh."
        
    audio, sr = librosa.load(audio_file)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
        
    result = predict_endpoint2(audio)
    return result

def predict_all(audio_file):
    result_1 = predict(audio_file)
    result_2 = predict2(audio_file)

    return result_1, result_2

# --- Giao diện Gradio ---
with gr.Blocks() as demo:
    gr.Markdown("## Audio Classification Interface")
    
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
        
    with gr.Row():
        predict_button = gr.Button("Predict (Dự đoán)", variant="primary") 
    with gr.Row():
        with gr.Column():
            output_text_1 = gr.Textbox(label="PhoWhisperSmall 86.00 result:")
        with gr.Column():
            output_text_2 = gr.Textbox(label="smart turn v3 pipe cat result:")
            
    predict_button.click(
        fn=predict_all, 
        inputs=[audio_input], 
        outputs=[output_text_1, output_text_2]
    )

demo.launch(share = True)