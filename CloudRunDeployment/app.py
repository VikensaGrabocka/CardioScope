# app.py
# Heart PCG Classification API
# Flask backend with VGG16 + DenseNet121, Grad-CAM, BPM detection, waveform visualizations

import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")  # no display needed in container
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.signal import butter, sosfilt, medfilt, find_peaks, hilbert
from kymatio.numpy import Scattering1D
from torchvision.models import vgg16, densenet121
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────
# Constants — identical to training code
# ─────────────────────────────────────────────
SR           = 4000
DURATION     = 5.0
TARGET_LEN   = int(SR * DURATION)   # 20000

BP_LOW       = 20
BP_HIGH      = 900
BP_ORDER     = 4
NOISE_ALPHA  = 0.005
MED_K        = 2
EPS          = 1e-9

WST_J        = 5
WST_Q        = 2

MFCC_N_MELS  = 60
MFCC_FMIN    = 20
MFCC_FMAX    = 1500
MFCC_FFT     = 512
MFCC_HOP     = 48
MFCC_WIN     = 192

IMG_H        = 256
IMG_W        = 256

CLASS_NAMES  = ["AS", "MR", "MS", "MVP", "N"]
CLASS_DESCRIPTIONS = {
    "AS":  "Aortic Stenosis",
    "MR":  "Mitral Regurgitation",
    "MS":  "Mitral Stenosis",
    "MVP": "Mitral Valve Prolapse",
    "N":   "Normal",
}
NUM_CLASSES  = 5

# ─────────────────────────────────────────────
# Bandpass filter — built once at startup
# ─────────────────────────────────────────────
def _butter_bandpass_sos(low, high, fs, order=4):
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")

_BPF_SOS = _butter_bandpass_sos(BP_LOW, BP_HIGH, SR, BP_ORDER)


# ─────────────────────────────────────────────
# Model architecture helper
# ─────────────────────────────────────────────
def build_classifier_head(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
        nn.Linear(1024, 512),         nn.ReLU(inplace=True), nn.Dropout(0.25),
        nn.Linear(512, NUM_CLASSES),
    )


# ─────────────────────────────────────────────
# Preprocessing — exact copy from training
# ─────────────────────────────────────────────
def load_and_preprocess_bytes(audio_bytes: bytes) -> np.ndarray:
    signal, _ = librosa.load(
        io.BytesIO(audio_bytes), sr=SR, mono=True, duration=DURATION
    )

    # Amplitude normalisation
    max_amp = np.max(np.abs(signal))
    if max_amp > 0:
        signal = signal / (max_amp + EPS)

    # Bandpass filter
    signal = sosfilt(_BPF_SOS, signal)

    # Adaptive noise gate
    thr = NOISE_ALPHA * np.max(np.abs(signal))
    signal = np.where(np.abs(signal) >= thr, signal, 0.0)

    # NaN/Inf correction
    signal = np.where(np.isfinite(signal), signal, 0.0)

    # Median filter
    signal = medfilt(signal, kernel_size=2 * MED_K + 1)

    # Final renormalisation
    max_amp = np.max(np.abs(signal))
    if max_amp > 0:
        signal = signal / (max_amp + EPS)
    else:
        signal = np.zeros_like(signal)

    # Trailing silence removal
    nonzero_idx = np.where(np.abs(signal) > 1e-6)[0]
    if len(nonzero_idx) > 0:
        signal = signal[: nonzero_idx[-1] + 1]

    # Pad or truncate to TARGET_LEN
    if len(signal) < TARGET_LEN:
        signal = np.pad(signal, (0, TARGET_LEN - len(signal)))
    else:
        signal = signal[:TARGET_LEN]

    return signal.astype(np.float32)


# ─────────────────────────────────────────────
# Feature extraction — exact copy from training
# ─────────────────────────────────────────────
def extract_wst(signal: np.ndarray) -> np.ndarray:
    scattering = Scattering1D(J=WST_J, shape=TARGET_LEN, Q=WST_Q)
    Sx = scattering(signal)
    Sx = Sx[Sx.std(axis=1) > 1e-10]
    Sx = np.log1p(np.abs(Sx))
    mu, sigma = Sx.mean(), Sx.std()
    if sigma > 0:
        Sx = (Sx - mu) / sigma
    return Sx.astype(np.float32)


def extract_mfcc(signal: np.ndarray) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=SR,
        n_mels=MFCC_N_MELS, fmin=MFCC_FMIN, fmax=MFCC_FMAX,
        n_fft=MFCC_FFT, hop_length=MFCC_HOP, win_length=MFCC_WIN,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    p1  = np.percentile(log_mel, 1)
    p99 = np.percentile(log_mel, 99)
    log_mel = np.clip(log_mel, p1, p99)
    denom = (log_mel.max() - log_mel.min()) + EPS
    return ((log_mel - log_mel.min()) / denom).astype(np.float32)


def fuse_wst_mfcc(wst: np.ndarray, mfcc: np.ndarray) -> np.ndarray:
    def _bicubic_resize(arr):
        t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(IMG_H, IMG_W),
                          mode="bicubic", align_corners=False)
        return t.squeeze()

    def _zscore(t):
        mu, sigma = t.mean(), t.std()
        return (t - mu) / (sigma + EPS)

    fused = (_zscore(_bicubic_resize(wst)) + _zscore(_bicubic_resize(mfcc))) / 2.0
    fused = torch.clamp(fused, -3.0, 3.0)
    f_min, f_max = fused.min(), fused.max()
    return ((fused - f_min) / ((f_max - f_min) + EPS)).numpy().astype(np.float32)


# ─────────────────────────────────────────────
# Heart Rate (BPM) Detection
# ─────────────────────────────────────────────
def estimate_bpm(signal: np.ndarray, sr: int = SR):
    """
    Estimate BPM from PCG signal using Hilbert envelope + peak detection.
    S1 (lub) sounds are the dominant peaks in a heart sound recording.
    Returns (bpm, peak_indices).
    """
    # Amplitude envelope via Hilbert transform
    analytic = hilbert(signal)
    envelope = np.abs(analytic)

    # Smooth envelope with 50ms window
    smooth_len = int(sr * 0.05)
    envelope   = np.convolve(envelope,
                             np.ones(smooth_len) / smooth_len,
                             mode="same")

    # Peaks: minimum 0.3s apart (allows up to 200 BPM)
    min_distance = int(sr * 0.3)
    peaks, _     = find_peaks(
        envelope,
        distance=min_distance,
        height=envelope.mean() * 0.5
    )

    if len(peaks) < 2:
        return None, peaks

    intervals_seconds = np.diff(peaks) / sr
    bpm = 60.0 / np.mean(intervals_seconds)
    return round(float(bpm), 1), peaks


# ─────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        tensor  : (1, 3, 256, 256) with requires_grad=True
        Returns : (256, 256) numpy heatmap, values 0–1
        """
        self.model.zero_grad()
        output = self.model(tensor)
        score  = output[0, class_idx]
        score.backward()

        # Weight activations by mean gradient
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam)

        # Normalise
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to 256×256
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(IMG_H, IMG_W),
            mode="bicubic",
            align_corners=False,
        ).squeeze().numpy()

        return cam


def get_gradcam_layer(model_name: str, model: nn.Module) -> nn.Module:
    """Return the last conv layer for Grad-CAM hookup."""
    if model_name == "densenet121":
        return model.features.denseblock4.denselayer16.conv2
    elif model_name == "vgg16":
        return model.features[28]
    else:
        raise ValueError(f"No Grad-CAM layer defined for {model_name}")


# ─────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────
BG = "#0d1117"
GRID_COLOR = "#1e2530"
TEXT_COLOR = "#8b98a9"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_waveform_image(signal: np.ndarray, peaks: np.ndarray) -> str:
    """Standard waveform with detected beat markers."""
    time_axis = np.linspace(0, DURATION, len(signal))

    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(time_axis, signal, color="#00d4aa", linewidth=0.8, alpha=0.9)

    if len(peaks) > 0:
        ax.scatter(time_axis[peaks], signal[peaks],
                   color="#ff6b6b", s=25, zorder=5, label="S1 beats")

    ax.set_xlim(0, DURATION)
    ax.set_xlabel("Time (s)", color=TEXT_COLOR, fontsize=8)
    ax.set_ylabel("Amplitude", color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3441")
    ax.grid(True, color=GRID_COLOR, linewidth=0.5)
    plt.tight_layout()
    return _fig_to_b64(fig)


def generate_spectrogram_image(fused_array: np.ndarray) -> str:
    """The exact 256×256 fused image the model sees."""
    fig, ax = plt.subplots(figsize=(3, 3), facecolor=BG)
    ax.imshow(fused_array, cmap="magma", aspect="auto", origin="lower")
    ax.set_title("Model input (WST + MFCC fusion)",
                 color=TEXT_COLOR, fontsize=8, pad=4)
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    return _fig_to_b64(fig)


def generate_saliency_waveform(
    signal: np.ndarray,
    cam_256: np.ndarray,
    peaks: np.ndarray,
) -> str:
    """
    Two-panel figure:
    Top    — waveform colored green→red by model attention
    Bottom — attention bar showing which time segments drove the decision
    """
    # Map 256 CAM columns → signal length via interpolation
    time_saliency_256  = cam_256.mean(axis=0)  # average over frequency → (256,)
    time_saliency_full = np.interp(
        np.linspace(0, 1, len(signal)),
        np.linspace(0, 1, len(time_saliency_256)),
        time_saliency_256,
    )

    time_axis = np.linspace(0, DURATION, len(signal))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 4.5), facecolor=BG,
        gridspec_kw={"hspace": 0.5, "height_ratios": [2, 1]},
    )

    # ── Top: waveform segments colored by saliency ──
    ax1.set_facecolor(BG)
    points   = np.array([time_axis, signal]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap="RdYlGn_r",
        norm=plt.Normalize(0, 1),
        linewidth=1.2,
        alpha=0.95,
    )
    lc.set_array(time_saliency_full[:-1])
    ax1.add_collection(lc)

    if len(peaks) > 0:
        ax1.scatter(time_axis[peaks], signal[peaks],
                    color="#ffffff", s=18, zorder=5, alpha=0.8)

    ax1.set_xlim(0, DURATION)
    ax1.set_ylim(signal.min() * 1.1, signal.max() * 1.1)
    ax1.set_title("PCG waveform — green = ignored  |  red = decisive",
                  color=TEXT_COLOR, fontsize=8, pad=5)
    ax1.set_ylabel("Amplitude", color=TEXT_COLOR, fontsize=7)
    ax1.tick_params(colors=TEXT_COLOR, labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#2a3441")
    ax1.grid(True, color=GRID_COLOR, linewidth=0.4)

    # Colorbar for saliency scale
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, orientation="vertical",
                        fraction=0.02, pad=0.01)
    cbar.set_label("Attention", color=TEXT_COLOR, fontsize=7)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    # ── Bottom: attention bar ──
    ax2.set_facecolor(BG)
    ax2.fill_between(time_axis, time_saliency_full,
                     alpha=0.85, color="#ff6b6b")
    ax2.plot(time_axis, time_saliency_full,
             color="#ff8c8c", linewidth=0.8, alpha=0.6)
    ax2.set_xlim(0, DURATION)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Model attention over time",
                  color=TEXT_COLOR, fontsize=8, pad=5)
    ax2.set_xlabel("Time (s)", color=TEXT_COLOR, fontsize=7)
    ax2.set_ylabel("Attention", color=TEXT_COLOR, fontsize=7)
    ax2.tick_params(colors=TEXT_COLOR, labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2a3441")
    ax2.grid(True, color=GRID_COLOR, linewidth=0.4)

    plt.tight_layout()
    return _fig_to_b64(fig)


def generate_gradcam_spectrogram(
    fused_array: np.ndarray,
    cam_256: np.ndarray,
) -> str:
    """Side-by-side: raw model input | Grad-CAM heatmap overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), facecolor=BG)

    axes[0].imshow(fused_array, cmap="magma", aspect="auto", origin="lower")
    axes[0].set_title("Model input", color=TEXT_COLOR, fontsize=8, pad=4)
    axes[0].axis("off")

    axes[1].imshow(fused_array, cmap="magma", aspect="auto", origin="lower")
    heatmap = axes[1].imshow(cam_256, cmap="jet", aspect="auto",
                              origin="lower", alpha=0.45)
    axes[1].set_title("Grad-CAM attention", color=TEXT_COLOR, fontsize=8, pad=4)
    axes[1].axis("off")

    # Colorbar for heatmap
    cbar = fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Attention", color=TEXT_COLOR, fontsize=7)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    plt.tight_layout(pad=0.8)
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# Load models once at startup
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[startup] Using device: {DEVICE}")


def _load_models():
    print("[startup] Loading VGG16 ...")
    vgg = vgg16(weights=None)
    vgg.classifier = build_classifier_head(vgg.classifier[0].in_features)
    vgg.load_state_dict(
        torch.load("models/vgg16.pth", map_location=DEVICE)
    )
    vgg.to(DEVICE).eval()

    print("[startup] Loading DenseNet121 ...")
    dense = densenet121(weights=None)
    dense.classifier = build_classifier_head(dense.classifier.in_features)
    dense.load_state_dict(
        torch.load("models/densenet121.pth", map_location=DEVICE)
    )
    dense.to(DEVICE).eval()

    print("[startup] Models ready.")
    return {"vgg16": vgg, "densenet121": dense}


MODELS = _load_models()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": list(MODELS.keys()),
        "device": str(DEVICE),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Send a .wav as 'audio'."}), 400

    audio_bytes = request.files["audio"].read()

    try:
        # ── 1. Preprocess signal ──
        signal = load_and_preprocess_bytes(audio_bytes)

        # ── 2. BPM estimation ──
        bpm, peaks = estimate_bpm(signal)

        # ── 3. Feature extraction ──
        wst   = extract_wst(signal)
        mfcc  = extract_mfcc(signal)
        fused = fuse_wst_mfcc(wst, mfcc)   # (256, 256)

        # ── 4. Build input tensor ──
        tensor_no_grad = (
            torch.from_numpy(fused).float()
            .unsqueeze(0).expand(3, -1, -1)   # (3, 256, 256)
            .unsqueeze(0).to(DEVICE)           # (1, 3, 256, 256)
        )

        # ── 5. Run both models (no grad for speed) ──
        results = {}
        with torch.no_grad():
            for name, model in MODELS.items():
                logits = model(tensor_no_grad)
                probs  = torch.softmax(logits, dim=1)[0]
                results[name] = {
                    "probs":      probs,
                    "confidence": float(probs.max()),
                    "class_idx":  int(probs.argmax()),
                }

        # ── 6. Pick the most confident model ──
        winner_name = max(results, key=lambda k: results[k]["confidence"])
        winner      = results[winner_name]
        loser_name  = [k for k in results if k != winner_name][0]
        loser       = results[loser_name]

        # ── 7. Grad-CAM on the winning model only ──
        winning_model = MODELS[winner_name]

        # Disable inplace ReLU — required for Grad-CAM backward hooks
        # (VGG16 uses inplace=True by default which breaks autograd views)
        for module in winning_model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        target_layer = get_gradcam_layer(winner_name, winning_model)
        gradcam      = GradCAM(winning_model, target_layer)

        tensor_grad = (
            torch.from_numpy(fused).float()
            .unsqueeze(0).expand(3, -1, -1).contiguous()
            .unsqueeze(0).to(DEVICE)
            .requires_grad_(True)
        )
        cam_256 = gradcam.compute(tensor_grad, winner["class_idx"])

        # Re-enable inplace ReLU for efficiency on next requests
        for module in winning_model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = True

        # ── 8. Generate visualizations ──
        waveform_img    = generate_waveform_image(signal, peaks)
        spectrogram_img = generate_spectrogram_image(fused)
        saliency_img    = generate_saliency_waveform(signal, cam_256, peaks)
        gradcam_img     = generate_gradcam_spectrogram(fused, cam_256)

        # Downsampled waveform points for native charting in the mobile app (500 pts)
        step = max(1, len(signal) // 500)
        waveform_points = signal[::step].tolist()

        # ── 9. Return full response ──
        predicted_class = CLASS_NAMES[winner["class_idx"]]
        return jsonify({
            # Prediction
            "predicted_class":   predicted_class,
            "predicted_label":   CLASS_DESCRIPTIONS[predicted_class],
            "confidence":        round(winner["confidence"], 4),
            "decided_by":        winner_name,
            "all_probabilities": {
                CLASS_NAMES[i]: round(float(winner["probs"][i]), 4)
                for i in range(NUM_CLASSES)
            },
            "model_votes": {
                winner_name: {
                    "class":      CLASS_NAMES[winner["class_idx"]],
                    "confidence": round(winner["confidence"], 4),
                },
                loser_name: {
                    "class":      CLASS_NAMES[loser["class_idx"]],
                    "confidence": round(loser["confidence"], 4),
                },
            },

            # Heart rate
            "bpm":            bpm,
            "beats_detected": int(len(peaks)),

            # Waveform data (for native chart rendering in app)
            "waveform_points":    waveform_points,

            # Visualizations (base64 PNG — display directly as images)
            "waveform_image":      waveform_img,      # standard waveform + beat markers
            "spectrogram_image":   spectrogram_img,   # fused WST+MFCC image
            "saliency_waveform":   saliency_img,      # waveform colored by Grad-CAM attention
            "gradcam_spectrogram": gradcam_img,       # spectrogram + heatmap overlay
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error":   str(e),
            "details": traceback.format_exc(),
        }), 500


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)