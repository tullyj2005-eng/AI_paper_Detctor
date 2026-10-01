import sys
from pathlib import Path
import pandas as pd

# This file is two folders below the project root, so add the root to the
# import path so that "detector" and "features" can be found.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr

from detector.predict import Detector

theme = gr.themes.Default().set(
    button_primary_background_fill="#b97920",
    button_primary_text_color="#ffffff",
    button_primary_background_fill_hover="#a06a1b",
)

human_color = "#1f77b4"
ai_color = "#a90000"

NICE_NAMES = {
    "avg_sentence_length": "Sentence length",
    "burstiness": "Burstiness",
    "comma_rate": "Commas",
    "proper_punctuation_usage": "Punctuation",
    "semicolon_rate": "Semicolons",
}

def split_bar(prob_ai):
    ai_pct = prob_ai * 100
    human_pct = 100 - ai_pct
    return f"""
    <div style="font-family: sans-serif; padding: 8px 4px;">
      <div style="display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 6px;">
        <span style="color: {human_color};">Human {human_pct:.0f}%</span>
        <span style="color: {ai_color};">AI {ai_pct:.0f}%</span>
      </div>
      <div style="display: flex; align-items: center; height: 28px;">
        <div style="flex: 1; height: 100%; background: #e4e3df; border-radius: 6px 0 0 6px;
                    display: flex; justify-content: flex-end; overflow: hidden;">
          <div style="width: {human_pct:.1f}%; background: {human_color};"></div>
        </div>
        <div style="width: 3px; height: 40px; background: #333;"></div>
        <div style="flex: 1; height: 100%; background: #e4e3df; border-radius: 0 6px 6px 0;
                    display: flex; justify-content: flex-start; overflow: hidden;">
          <div style="width: {ai_pct:.1f}%; background: {ai_color};"></div>
        </div>
      </div>
    </div>
    """

detector = Detector()
detector.threshold = 0.2


def predict(text):

    if not text.strip():
        return "<p>Paste some text first.</p>", "", None

    
    result = detector.predict(text)
    prob_ai = result["prob_ai"]
    if result["is_ai"]:
        verdict = f"### Flagged as AI\nThe model gives this a {prob_ai:.0%} chance of being AI-written."
    else:
        verdict = f"### Not flagged\nP(AI) is {prob_ai:.0%}, below the {result['threshold']:.0%} cut-off."
    if result["too_short"]:
        verdict += f"\n\n**Warning:** only {result['word_count']} words. Results are unreliable under 100."



    contrib = result["contributions"]
    df = pd.DataFrame({
        "feature": [NICE_NAMES.get(n, n) for n in contrib],
        "contribution": list(contrib.values()),
       })
    df["direction"] = ["toward AI" if v > 0 else "toward human" for v in df["contribution"]]

    
       


    return split_bar(prob_ai), verdict, df

feature_plot = gr.BarPlot(x="feature", y="contribution", color="direction",
    label="How each feature influenced this prediction")



with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# AI Essay Detector\nPaste an essay (100+ words works best) and click **Analyze**.")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Essay text", lines=12, placeholder="Paste essay text here...")
            submit_button = gr.Button("Analyze", variant="primary")
        with gr.Column():
            bar_output = gr.HTML()
            verdict_output = gr.Markdown()
            feature_plot = gr.BarPlot(
                x="feature", y="contribution", color="direction",
                label="How each feature influenced this prediction",
                color_map={"toward AI": ai_color, "toward human": human_color},
            )

    submit_button.click(
        predict,
        inputs=text_input,
        outputs=[bar_output, verdict_output, feature_plot],
    )

if __name__ == "__main__":
    demo.launch(show_error=True)
