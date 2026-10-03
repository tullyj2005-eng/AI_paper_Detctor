document.getElementById("analyze_button").addEventListener("click", analyze);



async function analyze() {
  const text = document.getElementById("essay_input").value;
  const result = document.getElementById("result");

  if (!text.trim()) {
    result.textContent = "Paste some text first.";
    return;
  }
  result.textContent = "Analyzing...";

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });
    const data = await response.json();

    const aiPct = Math.round(data.prob_ai * 100);
    const humanPct = 100 - aiPct;
    result.innerHTML = `AI: ${aiPct}%<br>Human: ${humanPct}%`;
  } catch (err) {
    result.textContent = "Can't reach the detector. Is server.py running?";
  }
}