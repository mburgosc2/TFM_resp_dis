import librosa
import soundfile as sf
import os

def augment_audio(audio_path, uuid, save_dir, sr=16000):
    """
    Aplica 4 transformaciones acústicas a un archivo de audio.
    Guarda los nuevos archivos .wav en el disco usando soundfile.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=10)
        augmented_items = []
        
        # Transformación 1: Pitch shift -2 semitones (Grave)
        y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
        fname1 = f"{uuid}_aug_pitch_down.wav"
        path1 = os.path.join(save_dir, fname1)
        sf.write(path1, y_pitch_down, sr)  # <-- Corregido
        augmented_items.append((f"{uuid}_aug_pitch_down", "pitch_down"))
        
        # Transformación 2: Pitch shift +2 semitones (Agudo)
        y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=+2)
        fname2 = f"{uuid}_aug_pitch_up.wav"
        path2 = os.path.join(save_dir, fname2)
        sf.write(path2, y_pitch_up, sr)  # <-- Corregido
        augmented_items.append((f"{uuid}_aug_pitch_up", "pitch_up"))
        
        # Transformación 3: Time stretch 0.9x (Lento)
        y_time_slow = librosa.effects.time_stretch(y, rate=0.9)
        fname3 = f"{uuid}_aug_time_slow.wav"
        path3 = os.path.join(save_dir, fname3)
        sf.write(path3, y_time_slow, sr)  # <-- Corregido
        augmented_items.append((f"{uuid}_aug_time_slow", "time_slow"))
        
        # Transformación 4: Time stretch 1.1x (Rápido)
        y_time_fast = librosa.effects.time_stretch(y, rate=1.1)
        fname4 = f"{uuid}_aug_time_fast.wav"
        path4 = os.path.join(save_dir, fname4)
        sf.write(path4, y_time_fast, sr)  # <-- Corregido
        augmented_items.append((f"{uuid}_aug_time_fast", "time_fast"))
        
        return augmented_items
    except Exception as e:
        print(f"Error procesando {uuid}: {e}")
        return []