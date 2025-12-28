import { pipeline } from "@xenova/transformers";
import { VectorCache, createCacheKey, CacheOptions } from "./cache.js";

let extractor: any;
let currentModel: string = "";

// Global cache instance (can be replaced via setGlobalCache)
let globalCache: VectorCache | null = null;

export interface BatchConfig {
  /** Maximum items per batch (default: 32) */
  maxBatchSize?: number;
  /** Maximum estimated tokens per batch (default: 8000) */
  maxTokensPerBatch?: number;
  /** Enable adaptive batch sizing based on text length (default: true) */
  adaptive?: boolean;
  /** Use cache for embeddings (default: true if cache exists) */
  useCache?: boolean;
  /** Progress callback for long operations */
  onProgress?: (processed: number, total: number) => void;
}

/**
 * Set the global vector cache instance.
 */
export function setGlobalCache(cache: VectorCache | null): void {
  globalCache = cache;
}

/**
 * Get the global vector cache instance.
 */
export function getGlobalCache(): VectorCache | null {
  return globalCache;
}

/**
 * Create a new cache and set it as global.
 */
export function createCache(options: CacheOptions = {}): VectorCache {
  globalCache = new VectorCache(options);
  return globalCache;
}

export async function getEmbedder(model: string = "Xenova/all-MiniLM-L6-v2") {
  if (!extractor || currentModel !== model) {
    extractor = await pipeline("feature-extraction", model);
    currentModel = model;
  }
  return extractor;
}

/**
 * Estimate token count for a text (rough approximation).
 * Uses ~4 characters per token heuristic.
 */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export async function embed(
  text: string,
  model?: string,
  useCache: boolean = true
): Promise<Float32Array> {
  const actualModel = model ?? "Xenova/all-MiniLM-L6-v2";
  
  // Check cache first
  if (useCache && globalCache) {
    const cacheKey = createCacheKey(text, actualModel);
    const cached = globalCache.get(cacheKey);
    if (cached) {
      return cached;
    }
  }
  
  const embedder = await getEmbedder(model);
  const output = await embedder(text, {
    pooling: "mean",
    normalize: true,
  });
  const vector = output.data as Float32Array;
  
  // Store in cache
  if (useCache && globalCache) {
    const cacheKey = createCacheKey(text, actualModel);
    globalCache.set(cacheKey, vector);
  }
  
  return vector;
}

/**
 * Batch embed multiple texts with dynamic batching and caching.
 * 
 * Features:
 * - Adaptive batch sizing based on text length
 * - Cache integration to skip already-embedded texts
 * - Progress callback for long operations
 */
export async function embedBatch(
  texts: string[],
  model?: string,
  config: BatchConfig = {}
): Promise<Float32Array[]> {
  const {
    maxBatchSize = 32,
    maxTokensPerBatch = 8000,
    adaptive = true,
    useCache = true,
    onProgress,
  } = config;
  
  const actualModel = model ?? "Xenova/all-MiniLM-L6-v2";
  const embedder = await getEmbedder(model);
  const results: (Float32Array | null)[] = new Array(texts.length).fill(null);
  
  // First pass: check cache and collect uncached texts
  const uncachedIndices: number[] = [];
  
  if (useCache && globalCache) {
    for (let i = 0; i < texts.length; i++) {
      const cacheKey = createCacheKey(texts[i], actualModel);
      const cached = globalCache.get(cacheKey);
      if (cached) {
        results[i] = cached;
      } else {
        uncachedIndices.push(i);
      }
    }
  } else {
    for (let i = 0; i < texts.length; i++) {
      uncachedIndices.push(i);
    }
  }
  
  // Report initial progress (cached items)
  const cachedCount = texts.length - uncachedIndices.length;
  if (onProgress && cachedCount > 0) {
    onProgress(cachedCount, texts.length);
  }
  
  // Second pass: embed uncached texts with dynamic batching
  let processed = cachedCount;
  let batchStart = 0;
  
  while (batchStart < uncachedIndices.length) {
    let batchEnd = batchStart;
    let batchTokens = 0;
    
    // Build batch with adaptive sizing
    while (batchEnd < uncachedIndices.length) {
      const idx = uncachedIndices[batchEnd];
      const textTokens = adaptive ? estimateTokens(texts[idx]) : 0;
      
      // Check if adding this text would exceed limits
      if (batchEnd > batchStart) {
        if (batchEnd - batchStart >= maxBatchSize) break;
        if (adaptive && batchTokens + textTokens > maxTokensPerBatch) break;
      }
      
      batchTokens += textTokens;
      batchEnd++;
    }
    
    // Process batch
    const batchIndices = uncachedIndices.slice(batchStart, batchEnd);
    const batchTexts = batchIndices.map(i => texts[i]);
    
    const outputs = await Promise.all(
      batchTexts.map((text) =>
        embedder(text, { pooling: "mean", normalize: true })
      )
    );
    
    // Store results and update cache
    for (let i = 0; i < batchIndices.length; i++) {
      const originalIdx = batchIndices[i];
      const vector = outputs[i].data as Float32Array;
      results[originalIdx] = vector;
      
      if (useCache && globalCache) {
        const cacheKey = createCacheKey(texts[originalIdx], actualModel);
        globalCache.set(cacheKey, vector);
      }
    }
    
    processed += batchIndices.length;
    if (onProgress) {
      onProgress(processed, texts.length);
    }
    
    batchStart = batchEnd;
  }

  return results as Float32Array[];
}

/**
 * Pre-warm the cache with known text-vector pairs.
 * Useful when loading from a snapshot.
 */
export function warmupCache(
  entries: Array<{ text: string; vector: Float32Array; model?: string }>
): void {
  if (!globalCache) {
    globalCache = new VectorCache();
  }
  
  for (const { text, vector, model } of entries) {
    const cacheKey = createCacheKey(text, model ?? "Xenova/all-MiniLM-L6-v2");
    globalCache.set(cacheKey, vector);
  }
}

/**
 * Clear the global cache.
 */
export function clearCache(): void {
  globalCache?.clear();
}

/**
 * Get cache statistics.
 */
export function getCacheStats() {
  return globalCache?.getStats() ?? { hits: 0, misses: 0, hitRate: 0, size: 0 };
}

/** Serialize Float32Array to base64 string for storage */
export function vectorToBase64(vector: Float32Array): string {
  const buffer = Buffer.from(vector.buffer);
  return buffer.toString("base64");
}

/** Deserialize base64 string back to Float32Array */
export function base64ToVector(base64: string): Float32Array {
  const buffer = Buffer.from(base64, "base64");
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
}
