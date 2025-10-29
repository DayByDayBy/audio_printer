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
import tempfile

# Import the DroneTrack class from drone.py
from simplistic_drone import DroneTrack

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
        temp_plot = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_plot.name, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        status = f"✅ Generated: {filename.name}\n📁 Saved to: {filename}"
        
        return str(filename), buf, status, str(filename) if autoplay else None
        
    except Exception as e:
        error_msg = f"❌ Error generating drone: {str(e)}"
        return None, None, error_msg, None


# Define presets (matching the guide)
PRESETS = [
    {
        "name": "Blade Runner Pad",
        "emoji": "🌃",
        "seed": 2001,
        "duration": 45,
        "voices": 6,
        "description": "Deep, ominous, cinematic atmosphere"
    },
    {
        "name": "Shimmer Wash",
        "emoji": "✨",
        "seed": 777,
        "duration": 40,
        "voices": 5,
        "description": "Bright, ethereal, spacious textures"
    },
    {
        "name": "Deep Rumble",
        "emoji": "🌊",
        "seed": 13,
        "duration": 50,
        "voices": 4,
        "description": "Sub-bass, dark, powerful resonance"
    },
    {
        "name": "Pulsing Energy",
        "emoji": "⚡",
        "seed": 808,
        "duration": 35,
        "voices": 7,
        "description": "Rhythmic, dynamic, hypnotic patterns"
    },
]


def load_preset(preset_index):
    """Load a preset by index."""
    preset = PRESETS[preset_index]
    status = f"✨ Loaded preset: {preset['name']}\n\n{preset['description']}"
    return preset["seed"], preset["duration"], preset["voices"], status


# Create Gradio interface
with gr.Blocks(title="Drone Synthesizer", theme=gr.themes.Soft()) as app:
    
    with gr.Tabs():
        # TAB 1: SYNTHESIZER
        with gr.Tab("🎛️ Synthesizer"):
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
                    
                    gr.Markdown("### ⚡ Quick Start Presets")
                    gr.Markdown("*Click a card to load classic sounds*")
                    
                    # Store preset buttons for later wiring
                    preset_buttons = []
                    
                    # Create preset cards
                    for i, preset in enumerate(PRESETS):
                        with gr.Group():
                            preset_btn = gr.Button(
                                f"{preset['emoji']} {preset['name']}", 
                                size="sm",
                                variant="secondary"
                            )
                            preset_buttons.append((preset_btn, i))
                            
                            gr.Markdown(
                                f"**Seed:** {preset['seed']} | "
                                f"**Duration:** {preset['duration']}s | "
                                f"**Voices:** {preset['voices']}\n\n"
                                f"*{preset['description']}*",
                                elem_classes="preset-description"
                            )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Output")
                    
                    status_output = gr.Textbox(
                        label="Status",
                        value="Ready to generate...",
                        interactive=False,
                        lines=3
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
            
            # NOW wire up preset buttons after status_output exists
            for preset_btn, idx in preset_buttons:
                preset_btn.click(
                    fn=lambda i=idx: load_preset(i),
                    outputs=[seed_input, duration_input, voices_input, status_output]
                )
            
            # Wire up the generate button
            generate_btn.click(
                fn=generate_drone,
                inputs=[username_input, seed_input, duration_input, voices_input, autoplay_input],
                outputs=[download_output, waveform_output, status_output, audio_output]
            )
        
        # TAB 2: USER GUIDE
        with gr.Tab("📖 User Guide"):
            # Load and display the HTML guide
            try:
                with open("index.html", "r", encoding="utf-8") as f:
                    guide_html = f.read()
                gr.HTML(guide_html)
            except FileNotFoundError:
                gr.Markdown(
                    """
                    # ⚠️ Guide Not Found
                    
                    The `index.html` guide file was not found in the same directory as this app.
                    
                    Please ensure `index.html` is in the same folder as `gradio_drone.py`.
                    """
                )
        
        # TAB 3: GETTING STARTED
        with gr.Tab("🚀 Getting Started"):
            gr.Markdown(
                """
                # 🚀 Getting Started with Drone Synthesizer
                
                ## 📦 Installation & Setup
                
                This project uses `uv` for modern Python package management.
                
                ### Option 1: Run Locally
                
                ```bash
                # Install uv if you don't have it
                curl -LsSf https://astral.sh/uv/install.sh | sh
                
                # Clone or download the project
                cd drone-gradio
                
                # Install dependencies
                uv sync
                
                # Run the app
                uv run gradio_drone.py
                ```
                
                The app will open in your browser at `http://localhost:7860`
                
                ### Option 2: Run on Local Network (Classroom Setup)
                
                If you want all students on the same WiFi to access the app:
                
                ```bash
                # Edit gradio_drone.py and change the last line to:
                app.launch(server_name="0.0.0.0", server_port=7860)
                
                # Then run it
                uv run gradio_drone.py
                ```
                
                Students can then access it at `http://YOUR_IP:7860` (find your IP with `ipconfig` on Windows or `ifconfig` on Mac/Linux)
                
                ### Option 3: Quick Demo with Public Link
                
                For a temporary public link (lasts 72 hours):
                
                ```bash
                # Edit gradio_drone.py and change the last line to:
                app.launch(share=True)
                
                # Run it - you'll get a public gradio.live URL
                uv run gradio_drone.py
                ```
                
                ## ☁️ Deploy to Hugging Face Spaces (Permanent Free Hosting)
                
                Hugging Face Spaces is perfect for hosting creative tools like this—no server maintenance required!
                
                ### Step 1: Create an account
                - Go to [huggingface.co](https://huggingface.co) and sign up (free)
                
                ### Step 2: Create a new Space
                - Click "New Space" from your profile
                - Name it something like "drone-synthesizer"
                - Choose "Gradio" as the Space SDK
                - Choose "Public" visibility
                - Click "Create Space"
                
                ### Step 3: Upload your files
                You need these files in your Space:
                - `app.py` (rename `gradio_drone.py` to `app.py`)
                - `drone.py` (the synthesis engine)
                - `index.html` (the guide)
                - `requirements.txt` (see below)
                
                ### Step 4: Create requirements.txt
                
                Create a file called `requirements.txt` with:
                ```
                gradio
                numpy
                scipy
                soundfile
                tqdm
                matplotlib
                ```
                
                ### Step 5: Push your code
                
                Either:
                - **Via web interface**: Drag and drop files into the Space's Files tab
                - **Via git**: Clone the Space repo and push like any git repository
                
                ```bash
                git clone https://huggingface.co/spaces/YOUR_USERNAME/drone-synthesizer
                cd drone-synthesizer
                # Copy your files here
                git add .
                git commit -m "Initial commit"
                git push
                ```
                
                ### Step 6: Wait for build
                The Space will automatically build and deploy. Once ready, you'll have a permanent URL like:
                `https://huggingface.co/spaces/YOUR_USERNAME/drone-synthesizer`
                
                ## 🎓 For Teachers
                
                ### Classroom Setup Recommendations
                
                **Best for most classrooms:**
                - Run locally on your computer with `server_name="0.0.0.0"`
                - Students access via your local IP
                - No internet required after initial setup
                - All generated files stay on your machine
                
                **Best for homework/independent work:**
                - Deploy to Hugging Face Spaces
                - Students can access from home
                - No local installation needed
                - Add a cleanup script if concerned about disk space
                
                ### Managing Generated Files
                
                Files accumulate in the `generated_drones/` folder. You can:
                - Keep them all as a class archive
                - Periodically move to a separate backup folder
                - Add auto-cleanup (delete files older than 7 days, etc.)
                
                ## 🐛 Troubleshooting
                
                **"Module not found" errors:**
                - Make sure you ran `uv sync` first
                - Check you're in the right directory
                
                **Port already in use:**
                - Change the port: `app.launch(server_port=7861)`
                - Or kill the other process using port 7860
                
                **Slow generation:**
                - Longer durations and more voices take more time
                - This is normal - it's rendering, not real-time
                
                **Can't hear anything:**
                - Some seeds produce very low frequencies
                - Try with headphones or better speakers
                - Check your system volume
                
                ## 📚 Project Structure
                
                ```
                drone-gradio/
                ├── gradio_drone.py    # This web interface
                ├── drone.py           # Synthesis engine
                ├── index.html         # User guide
                ├── pyproject.toml     # Dependencies (uv)
                ├── requirements.txt   # Dependencies (pip/HF)
                └── generated_drones/  # Output folder (created automatically)
                ```
                
                ## 🔗 Resources
                
                - **uv documentation**: [docs.astral.sh/uv](https://docs.astral.sh/uv/)
                - **Gradio documentation**: [gradio.app/docs](https://gradio.app/docs)
                - **Hugging Face Spaces**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
                
                ---
                
                **Questions?** Check the User Guide tab for synthesis concepts and sound recipes!
                """
            )


if __name__ == "__main__":
    app.launch(share=False)