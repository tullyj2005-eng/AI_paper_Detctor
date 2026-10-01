import sys
from pathlib import Path

# This file is two folders below the project root, so add the root to the
# import path so that "detector" and "features" can be found.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr

from detector.predict import Detector

detector = Detector()


def predict(text):
    result = detector.predict(text)
    prob_ai = result["prob_ai"]
    is_ai = result["is_ai"]
    return f"P(AI) = {prob_ai:.1%}  ->  {'AI' if is_ai else 'human'}"


with gr.Blocks() as demo:
    gr.Markdown("# AI Essay Detector")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Paste essay text here", lines=10)
            submit_button = gr.Button("Submit")
        with gr.Column():
            output_text = gr.Textbox(label="Prediction", lines=1)

    submit_button.click(predict, inputs=text_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch()
