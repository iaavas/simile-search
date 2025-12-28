import levenshtein from "fast-levenshtein";

export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

export function fuzzyScore(a: string, b: string): number {
  const dist = levenshtein.get(a.toLowerCase(), b.toLowerCase());
  const maxLen = Math.max(a.length, b.length);
  return 1 - dist / maxLen;
}

export function keywordScore(query: string, text: string): number {
  const q = query.toLowerCase().split(" ");
  const t = text.toLowerCase();
  const hits = q.filter((w) => t.includes(w)).length;
  return hits / q.length;
}
