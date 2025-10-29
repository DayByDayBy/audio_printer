"""
drone.py
---------

A tiny, seed‑driven, modular audio‑synthesis toolkit for “dirty ambient drones”.

Features
--------
* deterministic random generator (`np.random.default_rng(seed)`)
* basic waveforms: sine, saw, square, noise
* LFO for modulating any parameter
* ADSR envelope (linear segments)
* simple Moog‑style 24 dB/oct low‑pass filter
* soft‑clip distortion
* short convolution‑reverb (built‑in impulse response)
* master gain & normalisation
* render → numpy array → WAV file (float32 PCM)

All code is pure Python + NumPy/SciPy – no external DSP libraries required.
"""

from __future__ import annotations
import numpy as np
from scipy import signal
import soundfile as sf
from tqdm import tqdm
from typing import Callable, Tuple

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------


def seconds_to_samples(seconds: float, sr: int) -> int:
    return int(np.round(seconds * sr))


def db_to_linear(db: float) -> float:
    """Convert dB gain to linear factor."""
    return 10 ** (db / 20.0)


# ----------------------------------------------------------------------
# Oscillators
# ----------------------------------------------------------------------


class Oscillator:
    """Base class – all oscillators are callable:  osc(t) → waveform."""

    def __call__(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SineOsc(Oscillator):
    def __init__(self, freq: float):
        self.freq = freq

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * self.freq * t)


class SawOsc(Oscillator):
    def __init__(self, freq: float):
        self.freq = freq

    def __call__(self, t: np.ndarray) -> np.ndarray:
        # “ramp” –‑1 … 1 each period
        return 2.0 * (t * self.freq - np.floor(0.5 + t * self.freq))


class SquareOsc(Oscillator):
    def __init__(self, freq: float, duty: float = 0.5):
        self.freq = freq
        self.duty = duty

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return np.where(
            (t * self.freq) % 1 < self.duty, 1.0, -1.0
        )  # simple pulse wave


class WhiteNoise(Oscillator):
    """Noise is generated once per call – good for static textures."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=t.shape)


# ----------------------------------------------------------------------
# Modulators (LFOs, envelopes, etc.)
# ----------------------------------------------------------------------


class LFO:
    """Low‑frequency oscillator – can modulate any scalar parameter."""

    def __init__(
        self,
        rng: np.random.Generator,
        freq_range: Tuple[float, float] = (0.02, 0.2),
        depth_range: Tuple[float, float] = (0.0, 0.5),
        shape: Callable[[float], float] = np.sin,
    ):
        self.freq = rng.uniform(*freq_range)          # Hz
        self.depth = rng.uniform(*depth_range)        # proportion of modulation amount
        self.shape = shape                            # waveform of the LFO (sin by default)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        # Returns a modulation curve in the range [-depth, +depth]
        return self.depth * self.shape(2 * np.pi * self.freq * t)


class ADSR:
    """Linear ADSR envelope.  All times are in seconds."""

    def __init__(
        self,
        attack: float,
        decay: float,
        sustain_level: float,
        release: float,
        total_duration: float,
    ):
        self.attack = attack
        self.decay = decay
        self.sustain_level = sustain_level
        self.release = release
        self.total_duration = total_duration

        # sanity fix – ensure the envelope fits inside the total length
        max_env = attack + decay + release
        if max_env > total_duration:
            raise ValueError(
                f"ADSR segments ({max_env}s) exceed total duration ({total_duration}s)"
            )
        self.sustain = total_duration - max_env

    def __call__(self, t: np.ndarray) -> np.ndarray:
        env = np.zeros_like(t)

        # Attack
        a_end = self.attack
        a_mask = (t >= 0) & (t < a_end)
        env[a_mask] = (t[a_mask] / self.attack)

        # Decay
        d_start = a_end
        d_end = a_end + self.decay
        d_mask = (t >= d_start) & (t < d_end)
        env[d_mask] = 1.0 - (1.0 - self.sustain_level) * (
            (t[d_mask] - d_start) / self.decay
        )

        # Sustain (flat)
        s_start = d_end
        s_end = s_start + self.sustain
        s_mask = (t >= s_start) & (t < s_end)
        env[s_mask] = self.sustain_level

        # Release
        r_start = s_end
        r_end = r_start + self.release
        r_mask = (t >= r_start) & (t <= r_end)
        env[r_mask] = self.sustain_level * (
            1.0 - (t[r_mask] - r_start) / self.release
        )

        return env


# ----------------------------------------------------------------------
# Filters & effects
# ----------------------------------------------------------------------


def moog_lowpass(signal_in: np.ndarray, cutoff_hz: float, resonance: float, sr: int) -> np.ndarray:
    """
    Very simple 4‑stage Moog ladder approximation.
    Parameters
    ----------
    signal_in : np.ndarray
        Input audio (float32)
    cutoff_hz : float
        Desired cutoff (Hz).  Clipped to (0, sr/2)
    resonance : float
        0‑4 (0 = no resonance, 4 = self‑oscillation)
    sr : int
        Sample rate
    Returns
    -------
    np.ndarray
        Filtered audio
    """
    # Normalise frequency
    f = np.clip(cutoff_hz / (sr * 0.5), 0.0, 1.0)

    # Pre‑warp for bilinear transform (simplified)
    f = f * 1.16

    # One‑pole smoothing coefficient
    p = f * (1.0 - 0.8 * f)
    k = 2.0 * np.sin(f * np.pi * 0.5) - 1.0
    r = resonance * (1.0 - 0.5 * p)

    # State variables – 4 stages
    y1 = y2 = y3 = y4 = 0.0
    out = np.empty_like(signal_in)

    for i, x in enumerate(signal_in):
        x = x - r * y4  # feedback
        y1 = x * p + y1 * (1 - p)
        y2 = y1 * p + y2 * (1 - p)
        y3 = y2 * p + y3 * (1 - p)
        y4 = y3 * p + y4 * (1 - p)
        out[i] = y4

    return out


def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """
    Simple tanh‑based soft clipping.  `drive` >1 makes it dirtier.
    """
    return np.tanh(drive * x)


def short_reverb(signal_in: np.ndarray, sr: int, decay: float = 0.4) -> np.ndarray:
    """
    Very short convolution reverb using an exponential impulse response.
    The length is 0.5 s, enough to give space without turning the sound into a wash.
    """
    ir_len = seconds_to_samples(0.5, sr)
    t = np.linspace(0, ir_len / sr, ir_len, endpoint=False)
    ir = np.exp(-t / decay) * np.random.uniform(-1, 1, size=ir_len) * 0.2
    # Normalise IR to avoid overall gain change
    ir /= np.max(np.abs(ir)) + 1e-12
    return signal.convolve(signal_in, ir, mode="full")[: len(signal_in)]


# ----------------------------------------------------------------------
# Drone voice – everything together
# ----------------------------------------------------------------------


class DroneVoice:
    """
    A single “layer” of drone sound.  It is built from a primary oscillator (or noise)
    that can be frequency‑modulated, amplitude‑modulated, filtered and distorted.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        sr: int,
        duration: float,
        base_freq: float,
        waveform: str = "saw",
        noise_amount: float = 0.0,
    ):
        self.rng = rng
        self.sr = sr
        self.duration = duration
        self.base_freq = base_freq

        # Choose oscillator
        if waveform == "sine":
            self.osc = SineOsc(base_freq)
        elif waveform == "square":
            self.osc = SquareOsc(base_freq, duty=0.5)
        elif waveform == "saw":
            self.osc = SawOsc(base_freq)
        elif waveform == "noise":
            self.osc = WhiteNoise(rng)
        else:
            raise ValueError(f"Unknown waveform: {waveform}")

        # Optional extra noise that will be mixed after the main oscillator
        self.noise_amount = np.clip(noise_amount, 0.0, 1.0)
        if self.noise_amount > 0:
            self.noise = WhiteNoise(rng)

        # LFOs (frequency & amplitude)
        self.lfo_fm = LFO(rng, freq_range=(0.02, 0.2), depth_range=(0.0, 0.03))
        self.lfo_am = LFO(rng, freq_range=(0.03, 0.15), depth_range=(0.0, 0.4))

        # ADSR envelope – we keep a long sustain so the drone lives for the whole clip
        self.env = ADSR(
            attack=duration * 0.05,
            decay=duration * 0.05,
            sustain_level=0.6,
            release=duration * 0.1,
            total_duration=duration,
        )

        # Filter settings – cutoffs are randomised each voice
        self.cutoff = rng.uniform(200.0, 1500.0)  # Hz
        self.resonance = rng.uniform(0.1, 1.2)    # 0‑4 (but keep modest)

        # Distortion drive
        self.drive = rng.uniform(0.5, 2.0)

    def generate(self) -> np.ndarray:
        """Render the voice to a NumPy float32 array."""
        t = np.arange(seconds_to_samples(self.duration, self.sr), dtype=np.float32) / self.sr

        # ---- FM (frequency modulation) ----
        # The LFO depth is a proportion of the base frequency, yielding subtle pitch wobble
        fm = self.lfo_fm(t) * self.base_freq
        freq_t = self.base_freq + fm

        # Create the base oscillator with the time‑varying frequency
        # We use a small helper that re‑evaluates the oscillator each sample
        # (still fast enough for modest lengths)
        primary = np.empty_like(t)
        for i, ti in enumerate(t):
            # update oscillator frequency on the fly
            if isinstance(self.osc, SineOsc):
                primary[i] = np.sin(2 * np.pi * freq_t[i] * ti)
            elif isinstance(self.osc, SawOsc):
                primary[i] = 2.0 * (freq_t[i] * ti - np.floor(0.5 + freq_t[i] * ti))
            elif isinstance(self.osc, SquareOsc):
                phase = (freq_t[i] * ti) % 1.0
                primary[i] = 1.0 if phase < 0.5 else -1.0
            else:   # noise – ignore frequency modulation
                primary[i] = self.osc(ti)

        # Optional additional white‑noise layer
        if self.noise_amount > 0:
            primary = (1 - self.noise_amount) * primary + self.noise_amount * self.noise(t)

        # ---- AM (amplitude modulation) ----
        am = 1.0 + self.lfo_am(t)  # LFO centred at 1
        primary *= am

        # ---- Apply envelope ----
        primary *= self.env(t)

        # ---- Low‑pass filter ----
        filtered = moog_lowpass(primary, self.cutoff, self.resonance, self.sr)

        # ---- Distortion ----
        distorted = soft_clip(filtered, drive=self.drive)

        # ---- Final reverb (light) ----
        final = short_reverb(distorted, self.sr, decay=0.3)

        # Normalise voice internally to avoid clipping when voices are summed later
        max_abs = np.max(np.abs(final)) + 1e-12
        return final / max_abs


# ----------------------------------------------------------------------
# Whole drone‑track generator
# ----------------------------------------------------------------------


class DroneTrack:
    """
    A collection of independent `DroneVoice`s mixed together.
    The construction is fully defined by the seed → reproducible.
    """

    def __init__(
        self,
        seed: int | None = None,
        sr: int = 48000,
        duration: float = 30.0,
        num_voices: int = 4,
    ):
        self.rng = np.random.default_rng(seed)
        self.sr = sr
        self.duration = duration
        self.num_voices = num_voices

        # Master gain (in dB) – keep the final mix below 0 dB
        self.master_gain_db = -3.0

    def _choose_voice_params(self) -> dict:
        """Randomly pick parameters for a single voice."""
        # Frequency choice – we like low, muddy tones, but occasionally a higher harmonic
        base_freq = self.rng.choice([20, 30, 40, 50, 60, 80, 100, 120, 150, 200]) * self.rng.uniform(0.8, 1.2)

        waveform = self.rng.choice(["saw", "square", "sine", "noise"])
        noise_amount = self.rng.uniform(0.0, 0.2) if waveform != "noise" else 0.0

        return {
            "base_freq": base_freq,
            "waveform": waveform,
            "noise_amount": noise_amount,
        }

    def render(self, progress: bool = True) -> np.ndarray:
        """Render the complete track (float32, centred around 0)."""
        total_len = seconds_to_samples(self.duration, self.sr)
        mix = np.zeros(total_len, dtype=np.float32)

        voice_iter = range(self.num_voices)
        if progress:
            voice_iter = tqdm(voice_iter, desc="Rendering voices")

        for _ in voice_iter:
            vp = self._choose_voice_params()
            voice = DroneVoice(
                rng=self.rng,
                sr=self.sr,
                duration=self.duration,
                base_freq=vp["base_freq"],
                waveform=vp["waveform"],
                noise_amount=vp["noise_amount"],
            )
            mix += voice.generate()

        # Normalise the sum, then apply master gain
        max_abs = np.max(np.abs(mix)) + 1e-12
        mix = mix / max_abs
        mix *= db_to_linear(self.master_gain_db)

        # Optional final soft‑clip to tame any stray peaks introduced by summing
        mix = soft_clip(mix, drive=0.7)

        return mix.astype(np.float32)

    def write_wav(self, filename: str, progress: bool = True):
        audio = self.render(progress=progress)
        sf.write(filename, audio, self.sr, subtype="PCM_24")
        print(f"Wrote {filename} – {len(audio) / self.sr:.2f}s @ {self.sr} Hz")


# ----------------------------------------------------------------------
# Example usage (run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a reproducible ambient drone")
    parser.add_argument("-o", "--output", default="drone1.wav", help="output WAV file")
    parser.add_argument("-d", "--duration", type=float, default=30.0, help="seconds")
    parser.add_argument("-v", "--voices", type=int, default=7, help="number of independent layers")
    parser.add_argument("-s", "--seed", type=int, default=42, help="random seed (int) for reproducibility")
    args = parser.parse_args()

    track = DroneTrack(seed=args.seed, duration=args.duration, num_voices=args.voices)
    track.write_wav(args.output)
