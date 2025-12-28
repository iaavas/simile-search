import { HybridWeights } from "./types";

const DEFAULT_WEIGHTS: Required<HybridWeights> = {
  semantic: 0.7,
  fuzzy: 0.15,
  keyword: 0.15,
};

export function hybridScore(
  semantic: number,
  fuzzy: number,
  keyword: number,
  weights: HybridWeights = {}
): number {
  const w = { ...DEFAULT_WEIGHTS, ...weights };
  
  // Normalize weights to sum to 1
  const total = w.semantic + w.fuzzy + w.keyword;
  const normalizedSemantic = w.semantic / total;
  const normalizedFuzzy = w.fuzzy / total;
  const normalizedKeyword = w.keyword / total;

  return (
    normalizedSemantic * semantic +
    normalizedFuzzy * fuzzy +
    normalizedKeyword * keyword
  );
}

export function getDefaultWeights(): Required<HybridWeights> {
  return { ...DEFAULT_WEIGHTS };
}
