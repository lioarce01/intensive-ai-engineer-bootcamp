"""
Gradio Demo Template for AI/ML Projects

A professional template for creating interactive demos with Gradio.
Features:
- Clean UI with custom theme
- Error handling
- Examples
- Loading states
- Professional layout
"""

import gradio as gr
import torch
from transformers import pipeline
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Model Loading
# ============================================================================

@gr.cache  # Cache model to avoid reloading
def load_model():
    """Load ML model - only runs once"""
    try:
        logger.info("Loading model...")
        model = pipeline("sentiment-analysis")
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Load model globally
try:
    model = load_model()
except Exception as e:
    model = None
    logger.error(f"Failed to load model: {e}")

# ============================================================================
# Core Prediction Function
# ============================================================================

def predict(
    text: str,
    temperature: float = 0.7,
    max_length: int = 100,
    progress=gr.Progress()
) -> tuple[str, dict]:
    """
    Main prediction function with error handling and progress updates.

    Args:
        text: Input text to process
        temperature: Model temperature parameter
        max_length: Maximum length of output
        progress: Gradio progress tracker

    Returns:
        Tuple of (result_text, result_dict)
    """
    # Validate input
    if not text or not text.strip():
        return "⚠️ Please enter some text", {}

    if len(text) > 5000:
        return "⚠️ Text too long. Maximum 5000 characters.", {}

    try:
        # Update progress
        progress(0, desc="Starting...")

        # Model inference
        progress(0.3, desc="Processing with model...")

        if model is None:
            return "❌ Model not loaded. Please refresh the page.", {}

        result = model(text)[0]

        progress(0.7, desc="Formatting results...")

        # Format output
        output_text = f"""
        ## Analysis Complete ✅

        **Sentiment:** {result['label']}
        **Confidence:** {result['score']:.2%}

        ### Parameters Used:
        - Temperature: {temperature}
        - Max Length: {max_length}
        """

        output_dict = {
            result['label']: result['score'],
            f"Not {result['label']}": 1 - result['score']
        }

        progress(1.0, desc="Done!")

        return output_text, output_dict

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"❌ Error: {str(e)}", {}

# ============================================================================
# UI Definition
# ============================================================================

# Custom theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

# Build interface
with gr.Blocks(
    theme=theme,
    title="AI Demo Template",
    css="""
        .gradio-container {max-width: 1200px !important}
        .output-markdown h2 {color: #2563eb;}
    """
) as demo:

    # Header
    gr.Markdown("""
    # 🤖 AI Model Demo Template

    This is a professional template for showcasing your AI/ML models.
    Replace this with your project description.

    [![GitHub](https://img.shields.io/badge/GitHub-View_Code-blue?logo=github)](https://github.com/yourusername/yourrepo)
    """)

    with gr.Row():
        # Left column - Inputs
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input")

            input_text = gr.Textbox(
                label="Text Input",
                placeholder="Enter your text here...",
                lines=5,
                info="Maximum 5000 characters"
            )

            with gr.Accordion("⚙️ Advanced Options", open=False):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher values = more randomness"
                )

                max_length = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    label="Max Length",
                    info="Maximum output length"
                )

            predict_btn = gr.Button(
                "🚀 Run Prediction",
                variant="primary",
                size="lg"
            )

            clear_btn = gr.Button("🗑️ Clear", size="sm")

        # Right column - Outputs
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")

            output_text = gr.Markdown(
                label="Detailed Results",
                value="*Results will appear here...*"
            )

            output_label = gr.Label(
                label="Confidence Scores",
                num_top_classes=5
            )

    # Examples section
    gr.Markdown("### 💡 Try These Examples")

    gr.Examples(
        examples=[
            ["I absolutely love this product! It's amazing and works perfectly!", 0.7, 100],
            ["This is the worst experience I've ever had. Terrible service.", 0.7, 100],
            ["The movie was okay. Not great, not terrible, just average.", 0.7, 100],
        ],
        inputs=[input_text, temperature, max_length],
        outputs=[output_text, output_label],
        fn=predict,
        cache_examples=True,
        label="Click any example to try it"
    )

    # Info section
    with gr.Accordion("ℹ️ About This Demo", open=False):
        gr.Markdown("""
        ### Model Information
        - **Model:** Replace with your model name
        - **Task:** Sentiment Analysis (example)
        - **Framework:** PyTorch + Transformers

        ### How It Works
        1. Enter your text in the input box
        2. Adjust parameters if needed (optional)
        3. Click "Run Prediction"
        4. View results and confidence scores

        ### Limitations
        - Maximum input length: 5000 characters
        - English text only
        - Results may vary based on input quality

        ### Resources
        - [GitHub Repository](https://github.com/yourusername/yourrepo)
        - [Model Card](https://huggingface.co/model-name)
        - [Documentation](https://your-docs-link.com)

        ### Contact
        - GitHub: [@yourusername](https://github.com/yourusername)
        - Email: your.email@example.com
        """)

    # Event handlers
    predict_btn.click(
        fn=predict,
        inputs=[input_text, temperature, max_length],
        outputs=[output_text, output_label]
    )

    clear_btn.click(
        lambda: ("", 0.7, 100, "*Results will appear here...*", None),
        outputs=[input_text, temperature, max_length, output_text, output_label]
    )

    # Footer
    gr.Markdown("""
    ---
    Made with ❤️ using [Gradio](https://gradio.app) | [Report an Issue](https://github.com/yourusername/yourrepo/issues)
    """)

# ============================================================================
# Launch Configuration
# ============================================================================

if __name__ == "__main__":
    demo.queue(
        concurrency_count=3,  # Handle 3 requests simultaneously
        max_size=20  # Maximum queue size
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for temporary public URL
        show_error=True,
        show_api=True,  # Enable API documentation
        favicon_path=None,  # Add your favicon path
    )
