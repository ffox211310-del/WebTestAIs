// tiny-llm.js
// 超簡易LLM：文字単位の確率計算機
// UIに一切依存しない。これ単体で動く。

class TinyLLM {
  constructor({ dim = 8, order = 3, lr = 0.05 } = {}) {
    this.dim = dim;       // 埋め込みベクトルの次元
    this.order = order;   // 何個前まで見るか
    this.lr = lr;         // 学習率
    this.embed = new Map(); // token -> Float32Array
  }

  // トークン化（超簡易：1文字=1トークン）
  tokenize(text) {
    return text.split("");
  }

  // 埋め込み取得（なければランダム生成）
  getVec(token) {
    if (!this.embed.has(token)) {
      const v = new Float32Array(this.dim);
      for (let i = 0; i < this.dim; i++) v[i] = Math.random() * 2 - 1;
      this.embed.set(token, v);
    }
    return this.embed.get(token);
  }

  // 文脈ベクトル（直近order個の平均）
  contextVector(history) {
    const ctx = new Float32Array(this.dim);
    const n = Math.min(history.length, this.order);
    if (n === 0) return ctx;
    for (let i = history.length - n; i < history.length; i++) {
      const v = history[i];
      for (let d = 0; d < this.dim; d++) ctx[d] += v[d];
    }
    for (let d = 0; d < this.dim; d++) ctx[d] /= n;
    return ctx;
  }

  // 次のトークンの確率分布
  predict(history) {
    const ctx = this.contextVector(history);
    const scores = [];
    for (const [token, vec] of this.embed) {
      let dot = 0;
      for (let d = 0; d < this.dim; d++) dot += ctx[d] * vec[d];
      scores.push({ token, score: dot });
    }
    // softmax
    const max = Math.max(...scores.map(s => s.score));
    let sum = 0;
    for (const s of scores) {
      s.prob = Math.exp(s.score - max);
      sum += s.prob;
    }
    for (const s of scores) s.prob /= sum;
    scores.sort((a, b) => b.prob - a.prob);
    return scores;
  }

  // 学習（正解トークンを文脈方向に少し動かす）
  learn(text) {
    const tokens = this.tokenize(text);
    const history = [];
    for (const t of tokens) {
      if (history.length > 0) {
        const ctx = this.contextVector(history);
        const vec = this.getVec(t);
        for (let d = 0; d < this.dim; d++) {
          vec[d] += this.lr * (ctx[d] - vec[d]);
        }
      }
      history.push(this.getVec(t));
      if (history.length > this.order) history.shift();
    }
  }

  // サンプリング（温度付き）
  sample(scores, temperature = 1.0) {
    if (scores.length === 0) return null;
    const probs = scores.map(s => Math.pow(s.prob, 1 / temperature));
    const sum = probs.reduce((a, b) => a + b, 0);
    let r = Math.random() * sum;
    for (let i = 0; i < scores.length; i++) {
      r -= probs[i];
      if (r <= 0) return scores[i].token;
    }
    return scores[0].token;
  }

  // 生成ループ
  generate(prompt, length = 20, temperature = 0.8) {
    const tokens = this.tokenize(prompt);
    const history = tokens.map(t => this.getVec(t));
    const out = [...tokens];

    for (let i = 0; i < length; i++) {
      const scores = this.predict(history);
      const next = this.sample(scores, temperature);
      if (next === null) break;
      out.push(next);
      history.push(this.getVec(next));
      if (history.length > this.order) history.shift();
    }
    return out.join("");
  }

  stats() {
    return {
      vocab: this.embed.size,
      dim: this.dim,
      order: this.order,
    };
  }
}
