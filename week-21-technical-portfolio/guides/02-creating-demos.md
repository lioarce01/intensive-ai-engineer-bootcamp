# Creating Effective Demos for AI Projects

> Transform your AI projects into compelling, interactive demonstrations that showcase your technical skills.

## Table of Contents
- [Why Demos Matter](#why-demos-matter)
- [Gradio Demos](#gradio-demos)
- [Streamlit Demos](#streamlit-demos)
- [HuggingFace Spaces](#huggingface-spaces)
- [Video Demos](#video-demos)
- [Performance Optimization](#performance-optimization)

## Why Demos Matter

**Statistics**:
- Projects with live demos get **3x more stars** on GitHub
- **85% of hiring managers** prefer seeing working demos over just code
- Demo videos increase project engagement by **200%**

**What makes a great demo**:
1. **Works immediately** - No setup required
2. **Clear value** - Shows what problem it solves
3. **Interactive** - Users can try their own inputs
4. **Fast** - Responds in < 3 seconds
5. **Visually appealing** - Professional design
6. **Educational** - Shows how it works

## Gradio Demos

Gradio is perfect for ML model demos with minimal code.

### Basic Example

```python
import gradio as gr
from transformers import pipeline

# Load model once
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Analyze sentiment of input text"""
    if not text.strip():
        return "Please enter some text"

    result = classifier(text)[0]
    return f"Sentiment: {result['label']} (confidence: {result['score']:.2%})"

# Create interface
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        label="Input Text",
        placeholder="Enter text to analyze...",
        lines=3
    ),
    outputs=gr.Textbox(label="Result"),
    title="Sentiment Analysis Demo",
    description="Analyze the sentiment of any text using BERT",
    examples=[
        ["I love this product! It's amazing!"],
        ["This is terrible and disappointing."],
        ["The weather is okay today."]
    ],
)

if __name__ == "__main__":
    demo.launch()
```

### Advanced Gradio with Blocks

```python
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(
    prompt,
    max_length,
    temperature,
    top_p,
    top_k
):
    """Generate text continuation"""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Custom theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
)

# Build interface with Blocks
with gr.Blocks(theme=theme, title="Text Generation Demo") as demo:
    gr.Markdown("""
    # 🤖 GPT-2 Text Generation
    Generate creative text continuations using GPT-2.
    Enter a prompt and adjust parameters to control generation.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Prompt",
                placeholder="Once upon a time...",
                lines=5
            )

            with gr.Accordion("Generation Parameters", open=False):
                max_length = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=100,
                    step=10,
                    label="Max Length"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P"
                )
                top_k = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Top K"
                )

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="Generated Text",
                lines=10,
                show_copy_button=True
            )

    # Event handlers
    generate_btn.click(
        fn=generate_text,
        inputs=[input_text, max_length, temperature, top_p, top_k],
        outputs=output_text
    )

    # Examples
    gr.Examples(
        examples=[
            ["The future of artificial intelligence is", 100, 0.7, 0.9, 50],
            ["In a galaxy far far away", 150, 0.8, 0.95, 50],
            ["The key to happiness is", 80, 0.6, 0.85, 40],
        ],
        inputs=[input_text, max_length, temperature, top_p, top_k],
        label="Example Prompts"
    )

    gr.Markdown("""
    ### ℹ️ About
    This demo uses GPT-2, a 124M parameter language model.
    Adjust temperature for creativity (higher = more random).
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
```

### Gradio with Multiple Inputs/Outputs

```python
import gradio as gr
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def classify_image(image, labels_text):
    """
    Zero-shot image classification using CLIP
    """
    # Parse labels
    labels = [label.strip() for label in labels_text.split(",")]

    if not labels:
        return "Please provide at least one label", None

    # Process inputs
    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze()

    # Create results
    results = {label: float(prob) for label, prob in zip(labels, probs)}

    # Format output
    result_text = "## Classification Results\n\n"
    for label, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
        result_text += f"**{label}**: {prob:.1%}\n\n"

    return result_text, results

# Create interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ CLIP Zero-Shot Image Classifier")
    gr.Markdown("Upload an image and provide custom labels for classification.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            labels_input = gr.Textbox(
                label="Labels (comma-separated)",
                placeholder="dog, cat, bird, car",
                value="cat, dog, bird"
            )
            classify_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            text_output = gr.Markdown(label="Results")
            plot_output = gr.Label(label="Probabilities", num_top_classes=5)

    classify_btn.click(
        fn=classify_image,
        inputs=[image_input, labels_input],
        outputs=[text_output, plot_output]
    )

    gr.Examples(
        examples=[
            ["examples/cat.jpg", "cat, dog, bird"],
            ["examples/car.jpg", "car, truck, motorcycle, bicycle"],
        ],
        inputs=[image_input, labels_input]
    )

if __name__ == "__main__":
    demo.launch()
```

### Gradio Tips & Best Practices

1. **Use themes** for professional appearance
2. **Add examples** so users can try quickly
3. **Include descriptions** explaining what the demo does
4. **Show loading states** for long operations
5. **Handle errors gracefully** with clear messages
6. **Add parameter explanations** with `info` argument
7. **Use `gr.Accordion`** to hide advanced options
8. **Enable `show_copy_button`** for text outputs
9. **Add `gr.Examples`** with diverse test cases
10. **Use `queue()`** for handling concurrent requests

## Streamlit Demos

Streamlit is great for data-heavy applications and dashboards.

### Basic Example

```python
import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎭",
    layout="wide"
)

# Title and description
st.title("🎭 Sentiment Analysis Demo")
st.markdown("""
Analyze the sentiment of any text using state-of-the-art NLP models.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Choose Model",
        ["distilbert-base-uncased-finetuned-sst-2-english", "roberta-base-sentiment"]
    )

# Cache model loading
@st.cache_resource
def load_model(model):
    return pipeline("sentiment-analysis", model=model)

classifier = load_model(model_name)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area(
        "Enter text to analyze",
        height=150,
        placeholder="Type or paste your text here..."
    )

    if st.button("Analyze Sentiment", type="primary"):
        if text_input:
            with st.spinner("Analyzing..."):
                result = classifier(text_input)[0]

                st.success("Analysis complete!")

                # Display results
                st.metric(
                    label="Sentiment",
                    value=result['label'],
                    delta=f"{result['score']:.2%} confidence"
                )

                # Progress bar for confidence
                st.progress(result['score'])
        else:
            st.warning("Please enter some text")

with col2:
    st.info("""
    **How it works:**

    1. Enter your text
    2. Click "Analyze"
    3. Get instant results

    **Supported sentiments:**
    - Positive
    - Negative
    """)

# Examples
st.divider()
st.subheader("Try these examples:")

examples = [
    "I absolutely love this product!",
    "This is the worst experience ever.",
    "It's okay, nothing special."
]

cols = st.columns(len(examples))
for i, example in enumerate(examples):
    with cols[i]:
        if st.button(f"Example {i+1}", key=f"ex{i}"):
            st.session_state.text = example
            st.rerun()
```

### Advanced Streamlit with Multiple Pages

**app.py**:
```python
import streamlit as st

st.set_page_config(
    page_title="AI Demo Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Demo Suite")
st.markdown("""
A collection of AI models and tools for various tasks.
Select a demo from the sidebar.
""")

st.info("👈 Choose a demo from the sidebar to get started")

# Display stats
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Models", "5")

with col2:
    st.metric("Total Demos", "8")

with col3:
    st.metric("Users This Month", "1.2K")
```

**pages/1_🎭_Sentiment_Analysis.py**:
```python
import streamlit as st
from transformers import pipeline

st.title("🎭 Sentiment Analysis")

# Your sentiment analysis code here
```

**pages/2_🖼️_Image_Classification.py**:
```python
import streamlit as st
from PIL import Image

st.title("🖼️ Image Classification")

# Your image classification code here
```

### Streamlit Tips & Best Practices

1. **Use `@st.cache_resource`** for loading models
2. **Use `@st.cache_data`** for data processing
3. **Organize with columns** for better layout
4. **Add sidebar** for settings and navigation
5. **Use `st.spinner`** for loading states
6. **Show metrics** with `st.metric`
7. **Add expanders** for additional info
8. **Use session state** for multi-page apps
9. **Include progress bars** for long operations
10. **Add helpful tooltips** with `help` parameter

## HuggingFace Spaces

### Deploying to Spaces

**1. Create `app.py` with Gradio or Streamlit**

**2. Create `requirements.txt`**:
```
transformers>=4.30.0
torch>=2.0.0
gradio>=3.40.0
```

**3. Create `README.md` (Space Card)**:
```markdown
---
title: Your Demo Name
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 3.40.0
app_file: app.py
pinned: false
license: mit
---

# Your Demo Name

Description of what your demo does...

## Usage

Instructions for using the demo...

## Model

Information about the model used...
```

**4. Push to HuggingFace**:
```bash
# Install Git LFS
git lfs install

# Create space
huggingface-cli login
huggingface-cli repo create your-space-name --type space --space_sdk gradio

# Clone and push
git clone https://huggingface.co/spaces/username/your-space-name
cd your-space-name

# Add files
cp /path/to/app.py .
cp /path/to/requirements.txt .
cp /path/to/README.md .

# Commit and push
git add .
git commit -m "Initial commit"
git push
```

### Spaces Configuration

**For Gradio with GPU**:
```yaml
---
title: Your App
sdk: gradio
sdk_version: 3.40.0
app_file: app.py
pinned: false
license: mit
hardware: t4-small  # Options: cpu-basic, t4-small, t4-medium, a10g-small
---
```

**For Streamlit**:
```yaml
---
title: Your App
sdk: streamlit
sdk_version: 1.25.0
app_file: app.py
pinned: false
license: mit
---
```

### Environment Variables in Spaces

Create `.env` file (not committed):
```bash
OPENAI_API_KEY=your_key
HF_TOKEN=your_token
```

Access in code:
```python
import os
from huggingface_hub import hf_hub_download

# Use space secrets
api_key = os.environ.get("OPENAI_API_KEY")
```

Configure in Space settings:
1. Go to your Space settings
2. Add variables under "Repository secrets"

## Video Demos

### Recording Tools

**Screen Recording**:
- **OBS Studio** (Free, powerful)
- **Loom** (Easy, browser-based)
- **ScreenFlow** (Mac, professional)
- **Camtasia** (Windows/Mac, full-featured)

**GIF Creation**:
- **ScreenToGif** (Windows)
- **Kap** (Mac)
- **Peek** (Linux)
- **LICEcap** (Cross-platform)

### Video Structure

**1. Hook (0-5 seconds)**
- Show the result/value immediately
- "Watch this AI classify images in real-time"

**2. Demo (5-30 seconds)**
- Show the app in action
- Use real, interesting examples
- Highlight key features

**3. How It Works (30-60 seconds)**
- Brief technical overview
- Architecture diagram
- Key technologies

**4. Call to Action (60-90 seconds)**
- Link to GitHub
- Link to live demo
- Invite contributions

### Tips for Great Demo Videos

1. **Keep it short** - 60-90 seconds max
2. **Show, don't tell** - Let the demo speak
3. **Use real examples** - Not "test test 123"
4. **Add captions** - Many watch without sound
5. **Quality matters** - 1080p minimum
6. **Good pacing** - Not too fast, not too slow
7. **Background music** - Subtle and professional
8. **Clear call to action** - What should viewers do next?

### GIF Best Practices

```bash
# Optimize GIF size
gifsicle -O3 --colors 256 input.gif -o output.gif

# Convert video to GIF
ffmpeg -i input.mp4 -vf "fps=10,scale=800:-1:flags=lanczos" -c:v gif output.gif
```

**Tips**:
- Keep under 5 seconds
- Max size: 10MB (GitHub)
- Show one clear action
- Use in README above the fold

## Performance Optimization

### Model Loading

**Bad**:
```python
def predict(text):
    model = load_model()  # Loads every time!
    return model(text)
```

**Good (Gradio)**:
```python
model = load_model()  # Load once globally

def predict(text):
    return model(text)
```

**Good (Streamlit)**:
```python
@st.cache_resource
def load_model():
    return Model()

model = load_model()  # Cached
```

### Async Processing

**Gradio with Queue**:
```python
demo.queue(
    concurrency_count=3,  # Process 3 requests simultaneously
    max_size=20  # Max queue size
)
demo.launch()
```

**Gradio with Progress**:
```python
def long_operation(text, progress=gr.Progress()):
    progress(0, desc="Loading model...")
    model = load_model()

    progress(0.5, desc="Processing...")
    result = model(text)

    progress(1.0, desc="Done!")
    return result

gr.Interface(
    fn=long_operation,
    inputs=gr.Textbox(),
    outputs=gr.Textbox()
).launch()
```

### Caching Results

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(text):
    # This result will be cached
    return model(text)
```

### Resource Management

```python
import torch
import gc

def predict(text):
    with torch.no_grad():  # Disable gradient computation
        result = model(text)

    # Clear cache if needed
    torch.cuda.empty_cache()
    gc.collect()

    return result
```

## UI/UX Best Practices

### Design Principles

1. **Clarity over cleverness** - Make it obvious
2. **Fast feedback** - Show loading states
3. **Error handling** - Graceful failure messages
4. **Progressive disclosure** - Hide advanced options
5. **Consistent styling** - Use themes
6. **Responsive design** - Work on mobile
7. **Accessibility** - Screen reader support

### Example Error Handling

```python
def predict(text):
    try:
        if not text or not text.strip():
            return "⚠️ Please enter some text"

        if len(text) > 5000:
            return "⚠️ Text too long. Maximum 5000 characters."

        result = model(text)
        return f"✅ Result: {result}"

    except Exception as e:
        return f"❌ Error: {str(e)}. Please try again."
```

### Loading States

**Gradio**:
```python
def slow_function(text):
    yield "Processing... 0%"
    time.sleep(1)
    yield "Processing... 50%"
    time.sleep(1)
    yield "Done! 100%"

demo = gr.Interface(
    fn=slow_function,
    inputs=gr.Textbox(),
    outputs=gr.Textbox()
)
```

**Streamlit**:
```python
with st.spinner("Processing..."):
    result = expensive_function()
st.success("Done!")
```

## Checklist for Great Demos

- [ ] Works without login/API keys
- [ ] Responds in < 3 seconds
- [ ] Includes 3-5 example inputs
- [ ] Has clear instructions
- [ ] Shows loading states
- [ ] Handles errors gracefully
- [ ] Works on mobile
- [ ] Has professional design
- [ ] Includes "About" section
- [ ] Links to GitHub
- [ ] Shows model/tech info
- [ ] Has shareable URL
- [ ] Includes demo GIF/video
- [ ] Documented limitations
- [ ] Contact information

---

**Next**: [Deployment Strategies](03-deployment-strategies.md)
