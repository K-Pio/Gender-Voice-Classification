import torch
import numpy as np
import librosa

def _preprocess_audio_to_mfcc_batch(
    audio_path: str,
    sample_rate: int = 16_000,
    n_mfcc: int = 20,
    max_frames: int = 62,
) -> torch.Tensor:
    """
    Zwraca tensor wejściowy o kształcie (1, seq_len=62, input_size=20),
    zgodny z LSTM(batch_first=True).
    """
    # 1) Wczytanie audio mono + resampling do 16 kHz
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # 2) MFCC (kształt: n_mfcc x frames), domyślnie hop_length=512, n_fft=2048
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # 3) Dopasowanie długości do 62 ramek (przycięcie/padding zerami po osi czasu)
    if mfcc.shape[1] > max_frames:
        mfcc = mfcc[:, :max_frames]
    else:
        pad = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")

    # 4) (czas, cecha) = (62, 20), dodaj wymiar batch -> (1, 62, 20)
    x = torch.tensor(mfcc, dtype=torch.float32).T.unsqueeze(0)
    return x

@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    audio_path: str,
    device: str | torch.device = "cpu",
    label_names: list[str] = ("female", "male"),
):
    """
    Wykonuje predykcję dla jednego pliku audio.

    Zwraca:
      - pred_label (str): nazwa klasy
      - probs (torch.Tensor): tensor (num_classes,) z prawdopodobieństwami
      - logits_or_probs (torch.Tensor): surowe wyjście modelu (batch=1 usunięty)
    """
    model.eval()
    device = torch.device(device)

    # Preprocessing zgodny z Twoim datasetem
    x = _preprocess_audio_to_mfcc_batch(audio_path)  # (1, 62, 20)
    x = x.to(device)

    # Opcjonalna kontrola zgodności wejścia z modelem
    if hasattr(model, "lstm") and hasattr(model.lstm, "input_size"):
        assert model.lstm.input_size == x.size(-1), (
            f"input_size modelu ({model.lstm.input_size}) != liczba cech ({x.size(-1)})"
        )

    # Forward
    out = model(x)             # (1, num_classes), w Twojej architekturze już po Softmax
    out = out.squeeze(0)       # (num_classes,)

    # Jeśli model nie miałby Softmax, można by zastosować F.softmax tutaj.
    probs = out
    pred_idx = int(probs.argmax(dim=-1))
    pred_label = label_names[pred_idx] if 0 <= pred_idx < len(label_names) else str(pred_idx)

    return pred_label, probs.detach().cpu(), out.detach().cpu()
