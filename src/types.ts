export interface SearchItem<T = any> {
  id: string;
  text: string;
  metadata?: T;
}

export interface SearchResult<T = any> {
  id: string;
  text: string;
  score: number;
  metadata?: T;
  explain?: {
    semantic: number;
    fuzzy: number;
    keyword: number;
  };
}

export interface SearchOptions {
  topK?: number;
  explain?: boolean;
  filter?: (metadata: any) => boolean;
  /** Minimum score threshold (0-1). Results below this are filtered out */
  threshold?: number;
}

export interface HybridWeights {
  /** Semantic similarity weight (0-1), default: 0.7 */
  semantic?: number;
  /** Fuzzy string similarity weight (0-1), default: 0.15 */
  fuzzy?: number;
  /** Keyword match weight (0-1), default: 0.15 */
  keyword?: number;
}

export interface SimileConfig {
  /** Custom hybrid scoring weights */
  weights?: HybridWeights;
  /** Model to use for embeddings (default: "Xenova/all-MiniLM-L6-v2") */
  model?: string;
}

/** Serialized state for persistence */
export interface SimileSnapshot<T = any> {
  version: string;
  model: string;
  items: SearchItem<T>[];
  /** Base64-encoded Float32Array vectors */
  vectors: string[];
  createdAt: string;
}
