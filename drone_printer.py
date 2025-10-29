"""
gradio_drone.py
---------------
Gradio web interface for the drone synthesizer.
Provides a dashboard-style UI for generating ambient drones.
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import io

# Import the DroneTrack class from drone.py
from drone import DroneTrack

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("generated_drones")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_drone(username, seed, duration, voices, autoplay):
    """
    Generate a drone and return the audio file path, waveform plot, and status message.
    """
    if not username or username.strip() == "":
        username = "anonymous"
    
    username = username.strip().replace(" ", "_")
    
    # Generate timestamp
    TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = OUTPUT_DIR / f"{username}_{TIME_STAMP}_drone.wav"
    
    # Generate the drone
    try:
        track = DroneTrack(seed=seed, duration=duration, num_voices=voices)
        audio_data = track.render(progress=False)
        track.write_wav(str(filename), progress=False)
        
        # Create waveform visualization
        fig, ax = plt.subplots(figsize=(12, 4))
        time_axis = np.linspace(0, duration, len(audio_data))
        ax.plot(time_axis, audio_data, linewidth=0.5, color='#3498db')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(f'Waveform: {filename.name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration)
        ax.set_ylim(-1, 1)
        plt.tight_layout()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        status = f"✅ Generated: {filename.name}\n📁 Saved to: {filename}"
        
        return str(filename), buf, status, str(filename) if autoplay else None
        
    except Exception as e:
        error_msg = f"❌ Error generating drone: {str(e)}"
        return None, None, error_msg, None


def create_preset_handler(preset_seed, preset_duration, preset_voices, preset_name):
    """Return a function that loads preset values."""
    def handler():
        return preset_seed, preset_duration, preset_voices, f"Loaded preset: {preset_name}"
    return handler


# Define presets (matching the guide)
PRESETS = [
    ("Blade Runner Pad", 2001, 45, 6),
    ("Shimmer Wash", 777, 40, 5),
    ("Deep Rumble", 13, 50, 4),
    ("Pulsing Energy", 808, 35, 7),
]


# Create Gradio interface
with gr.Blocks(title="Drone Synthesizer", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🎛️ Drone Synthesizer
        ### A Sound Printer for Ambient Drones
        
        This is not a real-time instrument—it's a **sound printer**. 
        Design your sound carefully, then generate it. Think, then create.
        
        Each seed creates a completely unique and reproducible sound. 
        Share seeds with friends like sharing recipes!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Your Info")
            username_input = gr.Textbox(
                label="Username",
                placeholder="Enter your name",
                value="",
                info="Used in the output filename"
            )
            
            gr.Markdown("### 🎚️ Sound Parameters")
            
            seed_input = gr.Number(
                label="Seed",
                value=42,
                precision=0,
                info="The DNA of your sound (any integer)"
            )
            
            duration_input = gr.Slider(
                label="Duration (seconds)",
                minimum=10,
                maximum=120,
                value=30,
                step=5,
                info="How long the sound lasts"
            )
            
            voices_input = gr.Slider(
                label="Voices",
                minimum=1,
                maximum=12,
                value=5,
                step=1,
                info="Number of layered sounds (more = richer)"
            )
            
            autoplay_input = gr.Checkbox(
                label="Autoplay when ready",
                value=False,
                info="Play immediately after generation"
            )
            
            generate_btn = gr.Button("🎵 Generate Drone", variant="primary", size="lg")
            
            gr.Markdown("### ⚡ Quick Presets")
            gr.Markdown("*Load classic sounds to get started*")
            
            preset_buttons = []
            for preset_name, preset_seed, preset_duration, preset_voices in PRESETS:
                btn = gr.Button(preset_name, size="sm")
                preset_buttons.append((btn, preset_name, preset_seed, preset_duration, preset_voices))
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Output")
            
            status_output = gr.Textbox(
                label="Status",
                value="Ready to generate...",
                interactive=False,
                lines=2
            )
            
            waveform_output = gr.Image(
                label="Waveform Visualization",
                type="filepath"
            )
            
            audio_output = gr.Audio(
                label="Generated Audio",
                type="filepath",
                autoplay=False
            )
            
            download_output = gr.File(
                label="Download WAV File",
                visible=True
            )
    
    gr.Markdown(
        """
        ---
        ### 💡 Tips
        - **Start with presets** to hear what different parameters sound like
        - **Change one thing at a time** to understand each control
        - **Use meaningful seeds** - your birthday, favorite numbers, dates
        - **Think before generating** - this is deliberate sound design, not instant gratification
        - **All files auto-save** to the `generated_drones` folder - build your collection!
        
        ### 🧪 Experiment
        Try seed 1337, 9999, or today's date as YYYYMMDD. 
        Compare 3 voices vs 9 voices. Try 15 seconds vs 90 seconds.
        
        *The same seed always makes the same sound - share your discoveries!*
        """
    )
    
    # Wire up the generate button
    generate_btn.click(
        fn=generate_drone,
        inputs=[username_input, seed_input, duration_input, voices_input, autoplay_input],
        outputs=[download_output, waveform_output, status_output, audio_output]
    )
    
    # Wire up preset buttons
    for btn, name, seed, duration, voices in preset_buttons:
        btn.click(
            fn=lambda s=seed, d=duration, v=voices, n=name: (s, d, v, f"✨ Loaded preset: {n}"),
            outputs=[seed_input, duration_input, voices_input, status_output]
        )


if __name__ == "__main__":
    app.launch(share=False)