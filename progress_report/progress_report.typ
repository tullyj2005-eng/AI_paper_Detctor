#set document(title: "AI Text Detector — Progress Report")
#set page(paper: "us-letter", margin: (x: 0.85in, y: 0.75in), numbering: "1")
#set text(size: 10pt)
#set par(justify: true, leading: 0.58em, spacing: 0.9em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set text(size: 11.5pt)
#show heading: set block(above: 1.1em, below: 0.6em)
#set table(stroke: 0.4pt + luma(170), inset: (x: 5pt, y: 3pt))

#align(center)[
  #text(15pt, weight: "bold")[Tool for AI Detection in Academic Writing]\
  #v(0.2em)
  Checkpoint: Data Analysis & ML Approach — Progress Report\
  #text(style: "italic")[Joseph Tully, Tristan Moda, Sunny Kovvuri, Jonah Weber, David Saxton]\
  September 30, 2026
]

= Data

*Source.* We use the DAIGT v2 training set (Kaggle, `thedrcat/daigt-v2-train-dataset`): *44,868 essays*, each labeled human (0) or AI (1). It contains *27,371 human* and *17,497 AI* essays across *15 essay prompts* (e.g. "Car-free cities", "Exploring Venus"). The human essays are student argumentative essays, almost all from the PERSUADE corpus (25,996). The AI essays come from 16 generator sources, including Mistral-7B, Llama-2, GPT-3.5/4, Claude, PaLM, Falcon-180B and Cohere. No single generator makes up more than 14% of the AI class.

*Raw imbalances.* The full set is skewed in two ways that a classifier could exploit instead of learning writing style:
- *Length:* human essays average 418 words (SD 189, max 1,656). AI essays average 329 (SD 94) and are much more uniform in length.
- *Topic:* the human-to-AI ratio varies widely by prompt, from 6:1 for "Exploring Venus" (1,862 : 314) down to 1:2.3 for "Seeking multiple opinions" (1,552 : 3,624). A model could learn "this topic is usually AI".

*Working sample.* To reduce both shortcuts, `make_samples.py` (1) keeps only essays of 250–700 words, the range where the two length distributions overlap; (2) balances human and AI essays *within each prompt*; and (3) draws 2,000 essays (seed 0). The result has *1,008 human / 992 AI* essays with every prompt roughly 50/50. All analysis below uses this sample.

= Features & Exploratory Analysis

Each essay is turned into a small vector of style statistics computed by our own text-processing code (`features.py`). It includes a sentence splitter that handles abbreviations ("Dr.", "et al.", "Fig.") and numbered lists, covered by unit tests. The features are computed from raw text, so the same code will run on text typed into the demo.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, center, right),
    table.header([*Feature*], [*Human (mean ± SD)*], [*AI (mean ± SD)*], [*Higher in*], [*Single-feature AUC*]),
    [Commas per 1,000 words], [35.1 ± 19.8], [60.8 ± 26.9], [AI], [*0.81*],
    [Burstiness (CV of sentence length)], [0.460 ± 0.128], [0.358 ± 0.099], [Human], [*0.74*],
    [Word count (reference only)], [439 ± 116], [360 ± 77], [Human], [0.70],
    [Mean words per sentence], [21.6 ± 7.2], [18.9 ± 3.7], [Human], [0.61],
    [Vocabulary richness (unique / total)], [0.426 ± 0.070], [0.458 ± 0.085], [AI], [0.60],
    [Semicolons per 1,000 words], [0.48 ± 1.39], [0.08 ± 0.63], [Human], [0.56],
    [Proper sentence punctuation (share)], [0.934 ± 0.142], [0.952 ± 0.114], [—], [0.51],
  ),
  caption: [Feature statistics on the 2,000-essay sample. AUC = how well that one feature alone separates the classes (0.5 = chance, 1.0 = perfect), oriented so the higher class scores above 0.5.],
) <tab-features>

#figure(
  image("features.png", width: 100%),
  caption: [Distributions by class (blue = human, orange = AI); vertical lines are medians.],
) <fig-dist>

*Insights.*
+ *AI uses far more commas.* AI essays average 61 commas per 1,000 words against 35 for students, making it the strongest single feature (AUC 0.81). LLMs favor long clauses joined with commas and introductory phrases ("Additionally, ...", "In conclusion, ...").
+ *Human writing is "burstier".* Students mix very short and very long sentences. AI keeps sentence length steady: the SD of mean sentence length is about half the human value (3.7 vs 7.2) and burstiness is lower (0.36 vs 0.46). This supports the burstiness idea we started the project with.
+ *Semicolons are a human signal, but a rare one.* Most essays in both classes use none. Among those that do, nearly all are human. This is useful as a tie-breaker, weak on its own.
+ *Length still separates the classes after filtering.* Even within 250–700 words, human essays run about 80 words longer, and word count alone reaches AUC 0.70 (@fig-dist, right). We therefore use a *word-count-only baseline* that our style features must clearly beat.
+ *Not every planned feature helps.* "Proper punctuation" (sentence starts with a capital and ends in . ! ?) is close to 95% in both classes (AUC 0.51). Students in this corpus write mechanically clean sentences too, so we expect it to be dropped.

= ML Approach

*Model.* We use a scikit-learn pipeline: median imputation (some features are undefined for very short texts, e.g. burstiness needs at least 5 sentences) → standardization → *logistic regression*. We chose logistic regression as an interpretable first baseline: each coefficient shows how strongly a feature pushes toward "AI". The current model uses 5 features: burstiness, comma rate, semicolon rate, average sentence length and proper punctuation.

*Evaluation design: grouping by prompt.* A random split would put essays on the same topic in both train and test, letting the model score well by recognizing topics. Instead we split *by prompt*: 3 entire prompts (Car-free cities, Driverless cars, Facial action coding system; 513 essays) are held out as the test set. The other 12 prompts (1,487 essays) are used for 5-fold `StratifiedGroupKFold` cross-validation, which also groups by prompt. Every reported score is therefore on *topics the model has never seen*.

= Preliminary Results

#figure(
  table(
    columns: (auto, auto),
    align: (left, right),
    table.header([*Metric*], [*Value*]),
    [Cross-validated ROC-AUC, 5 style features], [*0.902* ± 0.031 (folds: .910 .919 .843 .907 .931)],
    [Cross-validated ROC-AUC, word count only (baseline)], [0.720 ± 0.113],
    [Held-out test ROC-AUC (3 unseen prompts)], [*0.920*],
    [Held-out accuracy at threshold 0.5], [81.3% (417 / 513)],
    [  False positive rate (human flagged as AI)], [29.8% (79 / 265)],
    [  Recall on AI essays], [93.1% (231 / 248)],
    [  Precision on AI predictions], [74.5%],
  ),
  caption: [Logistic regression results on the 2,000-essay sample.],
)

The style features beat the length-only baseline by about 0.18 AUC and generalize to unseen prompts: the held-out AUC of 0.92 is in line with cross-validation. The learned coefficients (standardized) agree with the exploratory analysis: comma rate +2.08 (pushes toward AI), burstiness −1.33, semicolon rate −0.69, average sentence length −0.42, proper punctuation −0.17 (all pushing toward human).

= Remaining Steps

+ *More features.* Add vocabulary richness (implemented, AUC 0.60, not yet in the model) and the planned transition-word density ("moreover", "furthermore"), type-token ratio on a fixed window, passive-voice ratio and reading level (`textstat`). Drop proper punctuation if it adds nothing.
+ *Model comparison.* Compare logistic regression with random forest and gradient boosting under the same prompt-grouped evaluation, tune hyperparameters, and use more of the 44k essays than the current 2,000.
+ *Error analysis.* Break results down by AI generator (e.g. Claude vs. Mistral) and look at which human essays get falsely flagged.
+ *Demo.* Save the trained pipeline (`joblib`) and connect it to our Tkinter demo (`demo.py`, window done). Typed text will go through the same `features.py` functions, and the demo will show each feature value next to the AI-confidence score.

*The default threshold is not usable for a real detector.* At 0.5 the model flags 30% of genuine student essays as AI. For a tool aimed at academic writing, a false accusation is the costlier error, so we also measured the detector at strict thresholds. At 1.1% false positives it catches 33% of AI essays. At 5.3% it catches 60%, and at 10.2% it catches 74%. The final demo will use a conservative threshold and report a confidence score rather than a hard verdict.
