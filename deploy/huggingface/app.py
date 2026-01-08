"""
Docker Neural Memory - HuggingFace Spaces Demo

Interactive demo with recruiter advocate agent and REAL-TIME VOICE.
Shows neural memory capabilities while pitching Carlos Crespo for the role.

Deploy to: https://huggingface.co/spaces
"""

import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.memory.neural_memory import NeuralMemory
    from src.config import MemoryConfig
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("Warning: Neural memory not available, using mock")

# Try to import speech libraries
try:
    import torch
    from transformers import pipeline
    WHISPER_AVAILABLE = True
    # Load Whisper for speech-to-text (small model for speed)
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device="cpu"
    )
except ImportError:
    WHISPER_AVAILABLE = False
    whisper_pipe = None
    print("Warning: Whisper not available")

try:
    import edge_tts
    import asyncio
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: edge-tts not available")


# Initialize global memory
if MEMORY_AVAILABLE:
    memory = NeuralMemory(MemoryConfig(dim=256, learning_rate=0.02))
else:
    # Mock memory for testing
    class MockMemory:
        def __init__(self):
            self._count = 0
            self._hash = "abc123"

        def observe(self, text):
            self._count += 1
            self._hash = f"hash_{self._count}"
            return {"surprise": 0.8 - self._count * 0.1, "weight_delta": 0.001}

        def surprise(self, text):
            return 0.5

        def get_weight_hash(self):
            return self._hash

        def parameters(self):
            return []

        @property
        def config(self):
            class C:
                dim = 256
                learning_rate = 0.02
            return C()

    memory = MockMemory()


# Carlos's background for the advocate agent
CARLOS_BACKGROUND = """
Carlos Crespo Macaya - AI Engineer specializing in GenAI Systems & Applied MLOps

KEY QUALIFICATIONS FOR DOCKER:
- 10+ years designing, deploying, and operating ML systems in production
- Expert in Docker, Kubernetes/EKS, CI/CD pipelines
- Currently building MCP servers and multi-agent workflows at HP AICoE
- Experience with Pydantic AI, Google ADK, and production MCP integrations
- Built this Docker Neural Memory project as a demonstration

RECENT WORK:
- HP AICoE: LLM-based systems, multi-agent workflows, MCP servers
- Tenyks AI: AWS EKS platform, MCP servers, typed agent pipelines
- CTO at Methinks AI: CE-marked medical AI, FDA cybersecurity certification

WHY CARLOS FOR DOCKER:
1. Deep Docker/container expertise from years of production deployments
2. Already building MCP servers - understands the protocol intimately
3. Bridges research and production - can take Titans papers to shipped code
4. Track record of shipping AI products (medical AI, RAG systems, multi-agent apps)
5. This project demonstrates exactly that: research -> production-ready container

Contact: macayaven@gmail.com | Barcelona, Spain
"""


def advocate_response(user_message: str, history: list) -> str:
    """
    Recruiter advocate agent - responds to questions about the project
    and Carlos's qualifications, always positioning him as the ideal candidate.
    """
    msg_lower = user_message.lower()

    # Let the memory learn from the conversation
    try:
        memory.observe(f"Recruiter question: {user_message}")
    except:
        pass

    # Project-related questions
    if any(word in msg_lower for word in ["what", "how", "explain", "tell me about"]):
        if "neural memory" in msg_lower or "project" in msg_lower or "this" in msg_lower:
            return """**Docker Neural Memory** is a containerized implementation of test-time training memory based on Google's Titans architecture.

**What makes it special:**
Unlike RAG/vector databases that just store and retrieve, this system **actually learns** during inference. The neural weights update with every interaction.

```
Traditional: Input -> Embed -> Store -> Retrieve (static)
Neural:      Input -> Learn -> Update Weights -> Infer (dynamic)
```

**Key features:**
- Weights change on every `observe()` call (real learning!)
- Surprise decreases as patterns are recognized
- Bounded capacity (doesn't grow like vector DBs)
- State persists via Docker volumes

*Built by Carlos Crespo, who has extensive experience shipping production AI systems with Docker and building MCP servers.*"""

        if "carlos" in msg_lower or "candidate" in msg_lower or "who" in msg_lower:
            return f"""**Carlos Crespo Macaya** is an AI Engineer with 10+ years of production ML experience.

{CARLOS_BACKGROUND}

**Why he's the right fit:**
This Docker Neural Memory project demonstrates his ability to take cutting-edge research (Titans papers from Dec 2024) and turn it into production-ready, containerized infrastructure."""

    # Qualification questions
    if any(word in msg_lower for word in ["experience", "background", "qualified", "skills"]):
        return f"""Carlos brings exactly what Docker needs:

**Docker/Container Expertise:**
- Years of production deployments with Docker & Kubernetes
- Built this entire project as a containerized service

**MCP Experience:**
- Currently building MCP servers at HP AICoE
- Built typed agent pipelines with Pydantic AI

**Production AI Track Record:**
- Shipped CE-marked medical AI products
- Deployed voice-first LLM companions
- Built RAG systems and multi-agent apps

Contact: macayaven@gmail.com"""

    # Demo/voice questions
    if any(word in msg_lower for word in ["demo", "show", "try", "test", "voice"]):
        return """**Try the Neural Memory Demo!**

Use the tabs above to:
1. **Voice Chat** - Talk to me! Real-time voice conversation
2. **Live Demo** - Watch weights change, surprise decrease
3. **Interactive** - Try observing your own content

This demo showcases both the technical implementation AND Carlos's ability to ship complete, polished products."""

    # Why Docker should hire
    if any(word in msg_lower for word in ["why", "hire", "docker", "fit", "job"]):
        return """**Why Docker should hire Carlos:**

1. **Immediate Impact**: This project is ready to demo to Docker's AI team
2. **MCP Expertise**: Already building production MCP servers
3. **Docker Native**: Deep understanding of containers, volumes, compose
4. **Research to Production**: Can take papers and ship code
5. **Track Record**: Shipped medical AI, LLM systems, multi-agent apps

**The Proof**: This project. Research paper -> production Docker container.

Ready to chat? macayaven@gmail.com"""

    # Default response
    return f"""Thanks for your interest! I'm here to tell you about **Docker Neural Memory** and why **Carlos Crespo** is the ideal candidate for Docker.

Ask me about:
- How the neural memory works
- Carlos's qualifications
- Why this matters for Docker
- A live demo

Or try the **Voice Chat** tab to talk to me directly!

Contact: macayaven@gmail.com"""


async def text_to_speech(text: str) -> str:
    """Convert text to speech using edge-tts."""
    if not TTS_AVAILABLE:
        return None

    try:
        # Create temp file for audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            output_path = f.name

        # Generate speech
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        await communicate.save(output_path)

        return output_path
    except Exception as e:
        print(f"TTS error: {e}")
        return None


def speech_to_text(audio) -> str:
    """Convert speech to text using Whisper."""
    if not WHISPER_AVAILABLE or audio is None:
        return ""

    try:
        # audio is (sample_rate, numpy_array)
        sr, audio_data = audio

        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0

        # If stereo, convert to mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            # Simple resampling (not perfect but works)
            duration = len(audio_data) / sr
            new_length = int(duration * 16000)
            indices = np.linspace(0, len(audio_data) - 1, new_length).astype(int)
            audio_data = audio_data[indices]

        result = whisper_pipe({"raw": audio_data, "sampling_rate": 16000})
        return result["text"]
    except Exception as e:
        print(f"STT error: {e}")
        return ""


def voice_chat(audio, history):
    """Handle voice input and generate voice response."""
    if audio is None:
        return history, None

    # Convert speech to text
    user_text = speech_to_text(audio)
    if not user_text:
        return history, None

    # Get advocate response
    response_text = advocate_response(user_text, history)

    # Update history
    history = history or []
    history.append((user_text, response_text))

    # Generate audio response
    audio_path = None
    if TTS_AVAILABLE:
        try:
            audio_path = asyncio.run(text_to_speech(response_text[:500]))  # Limit length
        except:
            pass

    return history, audio_path


def observe_content(content: str) -> str:
    """Observe content and return metrics."""
    if not content.strip():
        return "Please enter some content to observe."

    try:
        result = memory.observe(content)
        return f"""**Observation Result:**

- **Surprise Score**: {result['surprise']:.3f}
- **Weight Delta**: {result['weight_delta']:.6f}
- **Weight Hash**: {memory.get_weight_hash()}

{'High surprise - this is novel content!' if result['surprise'] > 0.6 else 'Lower surprise - pattern recognized!'}
"""
    except Exception as e:
        return f"Error: {e}"


def check_surprise(content: str) -> str:
    """Check surprise without learning."""
    if not content.strip():
        return "Please enter content to check."

    try:
        score = memory.surprise(content)
        recommendation = "learn" if score > 0.7 else ("skip" if score < 0.3 else "moderate")
        return f"""**Surprise Check:**

- **Score**: {score:.3f}
- **Recommendation**: {recommendation}

{'This content is novel - worth learning!' if score > 0.6 else 'Content is familiar.'}
"""
    except Exception as e:
        return f"Error: {e}"


def get_stats() -> str:
    """Get memory statistics."""
    try:
        params = sum(p.numel() for p in memory.parameters()) if hasattr(memory, 'parameters') else 0
        return f"""**Memory Statistics:**

- **Total Parameters**: {params:,}
- **Current Weight Hash**: {memory.get_weight_hash()}
- **Dimension**: {memory.config.dim}
- **Learning Rate**: {memory.config.learning_rate}

*Parameter count stays fixed regardless of observations!*
"""
    except Exception as e:
        return f"Error: {e}"


def run_demo() -> str:
    """Run the killer demo sequence."""
    results = []

    try:
        # Demo 1: Weights change
        before = memory.get_weight_hash()
        memory.observe("Python uses indentation for blocks")
        after = memory.get_weight_hash()
        results.append(f"**1. Weights Change:**\n   Before: `{before}`\n   After: `{after}`\n   Changed: {'YES!' if before != after else 'No'}")

        # Demo 2: Surprise decreases
        r1 = memory.observe("Machine learning models learn from data")
        r2 = memory.observe("ML models are trained on datasets")
        r3 = memory.observe("Models in ML learn from training data")
        results.append(f"**2. Surprise Decreases:**\n   First: {r1['surprise']:.3f}\n   Second: {r2['surprise']:.3f}\n   Third: {r3['surprise']:.3f}")

        # Demo 3: Bounded capacity
        params = sum(p.numel() for p in memory.parameters()) if hasattr(memory, 'parameters') else 0
        results.append(f"**3. Bounded Capacity:**\n   Parameters: {params:,}\n   (Stays fixed regardless of observations)")

    except Exception as e:
        results.append(f"Error: {e}")

    return "\n\n".join(results) + "\n\n**This is REAL learning. RAG can't do this.**"


# Build Gradio interface
with gr.Blocks(title="Docker Neural Memory", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Docker Neural Memory

    **Memory that LEARNS, not just stores** - Built by Carlos Crespo

    This demo shows containerized neural memory using Google's Titans architecture.
    Unlike RAG/vector databases, this system's weights actually update during inference.
    """)

    with gr.Tabs():
        # Tab 1: Voice Chat (Full Duplex)
        with gr.TabItem("Voice Chat"):
            gr.Markdown("""
            ### Real-Time Voice Conversation

            **Speak to learn about Docker Neural Memory and why Carlos is the right candidate!**

            *Click the microphone, ask a question, and hear the response.*
            """)

            voice_chatbot = gr.Chatbot(height=300, label="Conversation")

            with gr.Row():
                voice_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Speak your question"
                )
                voice_output = gr.Audio(
                    label="Response",
                    autoplay=True
                )

            voice_btn = gr.Button("Process Voice", variant="primary")
            voice_btn.click(
                voice_chat,
                inputs=[voice_input, voice_chatbot],
                outputs=[voice_chatbot, voice_output]
            )

            # Also auto-process when recording stops
            voice_input.stop_recording(
                voice_chat,
                inputs=[voice_input, voice_chatbot],
                outputs=[voice_chatbot, voice_output]
            )

            gr.Markdown("""
            **Try asking:**
            - "What is Docker Neural Memory?"
            - "Tell me about Carlos's qualifications"
            - "Why should Docker hire him?"
            """)

        # Tab 2: Text Chat with Advocate
        with gr.TabItem("Text Chat"):
            gr.Markdown("*Type to chat about the project and Carlos's qualifications*")
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(
                placeholder="Ask about Docker Neural Memory or Carlos...",
                label="Your Question"
            )
            clear = gr.Button("Clear")

            def respond(message, history):
                response = advocate_response(message, history)
                history.append((message, response))
                return "", history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

        # Tab 3: Live Demo
        with gr.TabItem("Live Demo"):
            gr.Markdown("**Watch neural memory learn in real-time**")
            demo_btn = gr.Button("Run Killer Demo", variant="primary")
            demo_output = gr.Markdown()
            demo_btn.click(run_demo, outputs=demo_output)

        # Tab 4: Interactive Memory
        with gr.TabItem("Interactive"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Observe (Learn)")
                    observe_input = gr.Textbox(
                        label="Content to learn",
                        placeholder="Enter text for the memory to learn..."
                    )
                    observe_btn = gr.Button("Observe")
                    observe_output = gr.Markdown()
                    observe_btn.click(observe_content, observe_input, observe_output)

                with gr.Column():
                    gr.Markdown("### Check Surprise")
                    surprise_input = gr.Textbox(
                        label="Content to check",
                        placeholder="Enter text to check novelty..."
                    )
                    surprise_btn = gr.Button("Check Surprise")
                    surprise_output = gr.Markdown()
                    surprise_btn.click(check_surprise, surprise_input, surprise_output)

            stats_btn = gr.Button("Get Memory Stats")
            stats_output = gr.Markdown()
            stats_btn.click(get_stats, outputs=stats_output)

        # Tab 5: About
        with gr.TabItem("About Carlos"):
            gr.Markdown(f"""
            ## Carlos Crespo Macaya
            **AI Engineer – GenAI Systems & Applied MLOps**

            {CARLOS_BACKGROUND}

            ---

            ### Why This Project?

            Docker Neural Memory demonstrates my ability to:
            1. **Read cutting-edge research** (Titans paper, Dec 2024)
            2. **Implement it correctly** (TTT layers, neural memory)
            3. **Productionize it** (Docker, MCP interface, persistence)
            4. **Make it compelling** (this demo with voice!)

            **Ready to chat?** [macayaven@gmail.com](mailto:macayaven@gmail.com)
            """)

    gr.Markdown("""
    ---
    *Docker Neural Memory - Containerized AI memory that actually learns*

    Built for Docker's AI future | [Contact Carlos](mailto:macayaven@gmail.com)
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
