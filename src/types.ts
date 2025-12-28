import { HNSWConfig } from "./ann.js";
import { CacheOptions, CacheStats } from "./cache.js";
import { QuantizationType } from "./quantization.js";
import { UpdaterConfig } from "./updater.js";

export { HNSWConfig, CacheOptions, CacheStats, QuantizationType, UpdaterConfig };

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
  /** Use fast similarity computation (default: true for large datasets) */
  useFastSimilarity?: boolean;
  /** Use ANN index if available (default: true) */
  useANN?: boolean;
}

export interface HybridWeights {
  /** Semantic similarity weight (0-1), default: 0.7 */
  semantic?: number;
  /** Fuzzy string similarity weight (0-1), default: 0.15 */
  fuzzy?: number;
  /** Keyword match weight (0-1), default: 0.15 */
  keyword?: number;
}

// Types moved to respective modules

// Types moved to respective modules

// Types moved to respective modules

// Types moved to respective modules

// ============ Simile Config Types ============

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
  /** Enable vector caching (default: true) */
  cache?: boolean | CacheOptions;
  /** Vector quantization type (default: 'float32') */
  quantization?: QuantizationType;
  /** Enable ANN index for large datasets (default: auto based on size) */
  useANN?: boolean | HNSWConfig;
  /** Minimum items to automatically enable ANN (default: 1000) */
  annThreshold?: number;
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
  /** Quantization type used */
  quantization?: QuantizationType;
  /** Serialized ANN index */
  annIndex?: any;
  /** Serialized cache */
  cache?: any;
}

// ============ Index Info Types ============

export interface IndexInfo {
  type: 'linear' | 'hnsw';
  size: number;
  memory: string;
  annStats?: {
    levels: number;
    avgConnections: number;
  };
  cacheStats?: CacheStats;
}
