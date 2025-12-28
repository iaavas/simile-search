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
    /** Raw scores before normalization */
    raw?: {
      semantic: number;
      fuzzy: number;
      keyword: number;
    };
  };
}

export interface SearchOptions {
  topK?: number;
  explain?: boolean;
  filter?: (metadata: any) => boolean;
  /** Minimum score threshold (0-1). Results below this are filtered out */
  threshold?: number;
  /** Minimum query length to trigger search (default: 1) */
  minLength?: number;
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
  /** 
   * Paths to extract searchable text from items. 
   * Supports nested paths like "author.firstName" or "tags[0]".
   * If not provided, uses the 'text' field directly.
   */
  textPaths?: string[];
  /** Whether to normalize scores across different scoring methods (default: true) */
  normalizeScores?: boolean;
}

/** Serialized state for persistence */
export interface SimileSnapshot<T = any> {
  version: string;
  model: string;
  items: SearchItem<T>[];
  /** Base64-encoded Float32Array vectors */
  vectors: string[];
  createdAt: string;
  /** Text paths used for extraction */
  textPaths?: string[];
}
