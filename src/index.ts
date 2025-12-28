export { Simile } from "./engine";
export * from "./types";
export { embed, embedBatch, vectorToBase64, base64ToVector } from "./embedder";
export { cosine, fuzzyScore, keywordScore, calculateScoreStats } from "./similarity";
export { hybridScore, getDefaultWeights } from "./ranker";
export { getByPath, extractText, normalizeScore } from "./utils";
