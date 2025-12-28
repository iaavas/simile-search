import levenshtein from "fast-levenshtein";

/**
 * Compute cosine similarity between two vectors.
 * Both vectors should be normalized (which they are from the embedder).
 * Returns a value between -1 and 1, where 1 is identical.
 */
export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/**
 * Compute fuzzy similarity score using Levenshtein distance.
 * Returns a value between 0 and 1, where 1 is an exact match.
 */
export function fuzzyScore(a: string, b: string): number {
  const aLower = a.toLowerCase();
  const bLower = b.toLowerCase();
  const dist = levenshtein.get(aLower, bLower);
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1;
  return 1 - dist / maxLen;
}

/**
 * Compute keyword match score.
 * Returns the proportion of query words found in the text (0 to 1).
 */
export function keywordScore(query: string, text: string): number {
  const queryWords = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (queryWords.length === 0) return 0;
  
  const textLower = text.toLowerCase();
  const hits = queryWords.filter((w) => textLower.includes(w)).length;
  return hits / queryWords.length;
}

/**
 * Score normalization statistics for a batch of results.
 */
export interface ScoreStats {
  semantic: { min: number; max: number };
  fuzzy: { min: number; max: number };
  keyword: { min: number; max: number };
}

/**
 * Calculate min/max statistics for score normalization.
 */
export function calculateScoreStats(
  scores: Array<{ semantic: number; fuzzy: number; keyword: number }>
): ScoreStats {
  if (scores.length === 0) {
    return {
      semantic: { min: 0, max: 1 },
      fuzzy: { min: 0, max: 1 },
      keyword: { min: 0, max: 1 },
    };
  }

  const stats: ScoreStats = {
    semantic: { min: Infinity, max: -Infinity },
    fuzzy: { min: Infinity, max: -Infinity },
    keyword: { min: Infinity, max: -Infinity },
  };

  for (const score of scores) {
    stats.semantic.min = Math.min(stats.semantic.min, score.semantic);
    stats.semantic.max = Math.max(stats.semantic.max, score.semantic);
    stats.fuzzy.min = Math.min(stats.fuzzy.min, score.fuzzy);
    stats.fuzzy.max = Math.max(stats.fuzzy.max, score.fuzzy);
    stats.keyword.min = Math.min(stats.keyword.min, score.keyword);
    stats.keyword.max = Math.max(stats.keyword.max, score.keyword);
  }

  return stats;
}
