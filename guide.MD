# Drone Synthesizer - Complete Setup Guide

## 📁 Project Structure

You'll need to create a new directory with these files:

```
drone-gradio/
├── gradio_drone.py       # The web interface (artifact provided)
├── drone.py              # The synthesis engine (your original file)
├── index.html            # The user guide (artifact provided)
├── pyproject.toml        # uv dependencies (see below)
├── requirements.txt      # pip/Hugging Face dependencies (see below)
├── README.md            # Project description (optional)
└── generated_drones/     # Created automatically when app runs
```

## 🛠️ Setup Steps

### 1. Create the Project Directory

```bash
mkdir drone-gradio
cd drone-gradio
```

### 2. Add Your Files

Copy the three main files:
- `drone.py` (your original synthesis engine)
- `gradio_drone.py` (from the artifact above)
- `index.html` (from the HTML artifact)

### 3. Create `pyproject.toml`

This is for `uv` package management:

```toml
[project]
name = "drone-gradio"
version = "0.1.0"
description = "Web interface for ambient drone synthesizer"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "gradio>=4.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "soundfile>=0.12.0",
    "tqdm>=4.65.0",
    "matplotlib>=3.7.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 4. Create `requirements.txt`

This is for Hugging Face Spaces (uses pip):

```txt
gradio>=4.0.0
numpy>=1.24.0
scipy>=1.10.0
soundfile>=0.12.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

## 🚀 Running Locally

### First Time Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run the app
uv run gradio_drone.py
```

The app will open at `http://localhost:7860`

### Subsequent Runs

```bash
uv run gradio_drone.py
```

## 🏫 Classroom Setup

To make the app accessible to all students on the same network:

1. Find your local IP address:
   - **Windows**: `ipconfig` (look for IPv4 Address)
   - **Mac/Linux**: `ifconfig` or `ip addr` (look for inet)

2. Edit `gradio_drone.py` - change the last line to:
   ```python
   app.launch(server_name="0.0.0.0", server_port=7860)
   ```

3. Run the app:
   ```bash
   uv run gradio_drone.py
   ```

4. Students access via: `http://YOUR_IP:7860`
   - Example: `http://192.168.1.42:7860`

## ☁️ Deploying to Hugging Face Spaces

### Why Hugging Face Spaces?

- **Free hosting** with generous resources
- **Zero maintenance** - no servers to manage
- **Permanent URL** - students can access from anywhere
- **Automatic builds** - just push code, it deploys
- **Perfect for creative tools** - not just ML!

### Deployment Steps

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co)

2. **Create a new Space:**
   - Go to your profile → "New Space"
   - Name: `drone-synthesizer` (or whatever you like)
   - License: Choose appropriate license (MIT is common)
   - SDK: Select **Gradio**
   - Visibility: Public (or Private if you prefer)
   - Click "Create Space"

3. **Prepare your files:**
   - Rename `gradio_drone.py` → `app.py` (Spaces expects this name)
   - Keep `drone.py` and `index.html` as-is
   - Use `requirements.txt` (not `pyproject.toml` - Spaces uses pip)

4. **Upload via Web Interface (Easiest):**
   - Go to your Space's "Files" tab
   - Drag and drop: `app.py`, `drone.py`, `index.html`, `requirements.txt`
   - The Space will automatically build

5. **OR Upload via Git (More Control):**
   ```bash
   # Clone your Space
   git clone https://huggingface.co/spaces/YOUR_USERNAME/drone-synthesizer
   cd drone-synthesizer
   
   # Copy your files
   cp /path/to/drone-gradio/gradio_drone.py app.py
   cp /path/to/drone-gradio/drone.py .
   cp /path/to/drone-gradio/index.html .
   cp /path/to/drone-gradio/requirements.txt .
   
   # Commit and push
   git add .
   git commit -m "Initial deployment"
   git push
   ```

6. **Wait for build:**
   - Watch the build logs in the Space
   - Should take 1-2 minutes
   - Once complete, your Space is live!

7. **Share the URL:**
   - `https://huggingface.co/spaces/YOUR_USERNAME/drone-synthesizer`
   - Students can bookmark this and use anytime

### Managing Storage on Spaces

Generated WAV files will accumulate. Options:

**Option A: Keep them (simple)**
- Spaces have decent storage
- Files are temporary anyway (cleared on rebuild)

**Option B: Auto-cleanup (recommended for long-term)**
Add this to `app.py` before `app.launch()`:

```python
import time
import os

def cleanup_old_files(max_age_hours=24):
    """Delete WAV files older than max_age_hours."""
    if not OUTPUT_DIR.exists():
        return
    
    now = time.time()
    for file in OUTPUT_DIR.glob("*.wav"):
        if now - file.stat().st_mtime > max_age_hours * 3600:
            file.unlink()

# Run cleanup on startup
cleanup_old_files(max_age_hours=24)
```

## 🎨 Customization Ideas

### Change the Theme

```python
with gr.Blocks(theme=gr.themes.Glass()) as app:  # or Monochrome, Soft, Base
```

### Add More Presets

Just expand the `PRESETS` list in `gradio_drone.py`:

```python
{
    "name": "Your Sound",
    "emoji": "🎵",
    "seed": 1234,
    "duration": 30,
    "voices": 5,
    "description": "Your description here"
}
```

### Change Default Parameters

Edit the default values in the input components:

```python
seed_input = gr.Number(
    label="Seed",
    value=42,  # Change this default
    precision=0,
)
```

## 📊 For Teachers: Usage Analytics

If deployed to Hugging Face Spaces, you can see:
- Number of visitors
- Usage patterns
- Space performance

Go to: Space Settings → Analytics

## 🐛 Common Issues

### "Module not found: drone"

Make sure `drone.py` is in the same directory as `gradio_drone.py` (or `app.py` on Spaces).

### "index.html not found"

The guide tab will show a warning but the app still works. Just add `index.html` to the same directory.

### Slow generation

This is normal! Rendering is not real-time:
- 30 seconds @ 5 voices ≈ 10-20 seconds to render
- 60 seconds @ 8 voices ≈ 30-40 seconds to render

### Port already in use

Change the port:
```python
app.launch(server_port=7861)  # or any other free port
```

## 📚 Learning Path for Students

1. **Start with presets** - understand what different sounds are possible
2. **Change one parameter** - see how seed, duration, voices affect output
3. **Read the guide** - understand what's happening under the hood
4. **Experiment systematically** - try sequential seeds, compare extremes
5. **Share discoveries** - what seeds do they find interesting?
6. **Read the code** - peek into `drone.py` to see how it works

## 🎯 Project Extensions

Ideas for advanced students:

- **Add new waveforms** to `drone.py`
- **Create a "favorites" system** using browser storage
- **Add visualization** of the frequency spectrum
- **Export settings as JSON** to share configurations
- **Add more effects** (delay, chorus, etc.)
- **Create "challenges"** - "make a sound that matches this description"

## 📖 Additional Resources

- **Gradio Documentation**: https://gradio.app/docs
- **NumPy for Audio**: https://numpy.org/doc/stable/
- **Sound Synthesis Basics**: Search for "Miller Puckette - Theory and Techniques of Electronic Music"
- **Hugging Face Spaces**: https://huggingface.co/docs/hub/spaces

---

## 🆘 Getting Help

If students have questions:
1. Check the "User Guide" tab in the app
2. Check the "Getting Started" tab for setup issues
3. Look at the code comments in `drone.py`
4. Experiment! - there are no wrong sounds

Happy droning! 🎛️✨