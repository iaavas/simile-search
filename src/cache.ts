/**
 * Vector Cache - LRU cache for embedding vectors with text hashing.
 * Avoids re-embedding duplicate or previously seen texts.
 */

export interface CacheOptions {
  /** Maximum number of entries to cache (default: 10000) */
  maxSize?: number;
  /** Enable hit/miss statistics tracking (default: false) */
  enableStats?: boolean;
}

export interface CacheStats {
  hits: number;
  misses: number;
  hitRate: number;
  size: number;
}

export interface SerializedCache {
  entries: Array<[string, string]>; // [hash, base64Vector]
  maxSize: number;
}

/**
 * MurmurHash3 - Fast, collision-resistant hash function.
 * Used for creating cache keys from text content.
 */
export function murmurHash3(str: string, seed: number = 0): string {
  let h1 = seed >>> 0;
  const c1 = 0xcc9e2d51;
  const c2 = 0x1b873593;

  for (let i = 0; i < str.length; i++) {
    let k1 = str.charCodeAt(i);
    
    k1 = Math.imul(k1, c1);
    k1 = (k1 << 15) | (k1 >>> 17);
    k1 = Math.imul(k1, c2);
    
    h1 ^= k1;
    h1 = (h1 << 13) | (h1 >>> 19);
    h1 = Math.imul(h1, 5) + 0xe6546b64;
  }

  h1 ^= str.length;
  h1 ^= h1 >>> 16;
  h1 = Math.imul(h1, 0x85ebca6b);
  h1 ^= h1 >>> 13;
  h1 = Math.imul(h1, 0xc2b2ae35);
  h1 ^= h1 >>> 16;

  return (h1 >>> 0).toString(16).padStart(8, '0');
}

/**
 * Create a cache key from text content.
 * Uses double hashing for better collision resistance.
 */
export function createCacheKey(text: string, model: string): string {
  const textHash = murmurHash3(text, 0);
  const modelHash = murmurHash3(model, 1);
  return `${textHash}-${modelHash}`;
}

/**
 * LRU (Least Recently Used) Vector Cache.
 * Provides O(1) get/set operations with automatic eviction.
 */
export class VectorCache {
  private cache: Map<string, Float32Array>;
  private maxSize: number;
  private enableStats: boolean;
  private hits: number = 0;
  private misses: number = 0;

  constructor(options: CacheOptions = {}) {
    this.maxSize = options.maxSize ?? 10000;
    this.enableStats = options.enableStats ?? false;
    this.cache = new Map();
  }

  /**
   * Get a cached vector by text content.
   * Returns undefined if not in cache.
   */
  get(key: string): Float32Array | undefined {
    const vector = this.cache.get(key);
    
    if (vector !== undefined) {
      // Move to end for LRU (delete and re-add)
      this.cache.delete(key);
      this.cache.set(key, vector);
      
      if (this.enableStats) this.hits++;
      return vector;
    }
    
    if (this.enableStats) this.misses++;
    return undefined;
  }

  /**
   * Cache a vector for a text content.
   */
  set(key: string, vector: Float32Array): void {
    // If key exists, delete first to update LRU order
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }
    
    // Evict oldest entries if at capacity
    while (this.cache.size >= this.maxSize) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey !== undefined) {
        this.cache.delete(oldestKey);
      }
    }
    
    this.cache.set(key, vector);
  }

  /**
   * Check if a key exists in cache.
   */
  has(key: string): boolean {
    return this.cache.has(key);
  }

  /**
   * Clear all cached entries.
   */
  clear(): void {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Get current cache size.
   */
  get size(): number {
    return this.cache.size;
  }

  /**
   * Get cache statistics.
   */
  getStats(): CacheStats {
    const total = this.hits + this.misses;
    return {
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? this.hits / total : 0,
      size: this.cache.size,
    };
  }

  /**
   * Reset statistics counters.
   */
  resetStats(): void {
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Serialize cache for persistence.
   */
  serialize(): SerializedCache {
    const entries: Array<[string, string]> = [];
    
    for (const [key, vector] of this.cache) {
      const buffer = Buffer.from(vector.buffer);
      entries.push([key, buffer.toString('base64')]);
    }
    
    return {
      entries,
      maxSize: this.maxSize,
    };
  }

  /**
   * Deserialize and restore cache from saved state.
   */
  static deserialize(data: SerializedCache, options: CacheOptions = {}): VectorCache {
    const cache = new VectorCache({
      maxSize: data.maxSize,
      ...options,
    });
    
    for (const [key, base64] of data.entries) {
      const buffer = Buffer.from(base64, 'base64');
      const vector = new Float32Array(
        buffer.buffer,
        buffer.byteOffset,
        buffer.length / 4
      );
      cache.cache.set(key, vector);
    }
    
    return cache;
  }

  /**
   * Pre-warm cache with existing vectors.
   */
  warmup(entries: Array<{ key: string; vector: Float32Array }>): void {
    for (const { key, vector } of entries) {
      this.set(key, vector);
    }
  }

  /**
   * Get all keys currently in cache.
   */
  keys(): string[] {
    return Array.from(this.cache.keys());
  }

  /**
   * Estimate memory usage in bytes.
   */
  getMemoryUsage(): number {
    let bytes = 0;
    for (const [key, vector] of this.cache) {
      bytes += key.length * 2; // UTF-16 string
      bytes += vector.byteLength;
    }
    return bytes;
  }
}
