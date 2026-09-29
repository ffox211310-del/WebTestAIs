// app.js
// UIとエンジンの接続だけを担当

const engine = new TinyLLM({ dim: 8, order: 3, lr: 0.05 });

const $ = id => document.getElementById(id);

// 学習
$("learnBtn").addEventListener("click", () => {
  const text = $("trainText").value;
  const iters = parseInt($("iters").value) || 1;

  const t0 = performance.now();
  for (let i = 0; i < iters; i++) engine.learn(text);
  const dt = performance.now() - t0;

  const s = engine.stats();
  $("status").textContent =
    `学習 ${dt.toFixed(1)}ms | 語彙 ${s.vocab} | dim ${s.dim} | order ${s.order}`;
});

// 生成
$("genBtn").addEventListener("click", () => {
  const prompt = $("prompt").value;
  const length = parseInt($("length").value) || 20;
  const temp = parseFloat($("temp").value) || 0.8;

  if (!prompt) {
    $("output").textContent = "(プロンプトを入れてください)";
    return;
  }

  const t0 = performance.now();
  const result = engine.generate(prompt, length, temp);
  const dt = performance.now() - t0;

  $("output").textContent = result;
  $("status").textContent = `生成 ${dt.toFixed(1)}ms | 温度 ${temp}`;
});

// リセット
$("resetBtn").addEventListener("click", () => {
  engine.embed.clear();
  $("output").textContent = "";
  $("status").textContent = "リセット完了";
});
