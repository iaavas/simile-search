import { embed, embedBatch, vectorToBase64, base64ToVector } from "./embedder.js";
import { cosine, fuzzyScore, keywordScore, calculateScoreStats } from "./similarity.js";
import { hybridScore, getDefaultWeights } from "./ranker.js";
import { extractText, normalizeScore } from "./utils.js";
import { 
  SearchItem, 
  SearchResult, 
  SearchOptions, 
  SimileConfig, 
  SimileSnapshot, 
  HybridWeights,
  QuantizationType,
  CacheStats,
  IndexInfo
} from "./types.js";
import { VectorCache, createCacheKey } from "./cache.js";
import { HNSWIndex } from "./ann.js";
import { BackgroundUpdater } from "./updater.js";
import { quantizeVector, dequantizeVector, QuantizedVector, base64ToQuantized, quantizedToBase64 } from "./quantization.js";


const PACKAGE_VERSION = "0.4.0";

export class Simile<T = any> {
  private items: SearchItem<T>[];
  private vectors: Float32Array[];
  private itemIndex: Map<string, number>;
  private config: Required<SimileConfig>;
  private cache: VectorCache | null = null;
  private annIndex: HNSWIndex | null = null;
  private updater: BackgroundUpdater<T>;

  private constructor(
    items: SearchItem<T>[],
    vectors: Float32Array[],
    config: SimileConfig = {}
  ) {
    this.items = items;
    this.vectors = vectors;
    this.itemIndex = new Map(items.map((item, i) => [item.id, i]));
    this.config = {
      weights: config.weights ?? getDefaultWeights(),
      model: config.model ?? "Xenova/all-MiniLM-L6-v2",
      textPaths: config.textPaths ?? [],
      normalizeScores: config.normalizeScores ?? true,
      cache: config.cache ?? true,
      quantization: config.quantization ?? 'float32',
      useANN: config.useANN ?? false,
      annThreshold: config.annThreshold ?? 1000,
    };

    // Initialize Cache
    if (this.config.cache) {
      this.cache = new VectorCache(typeof this.config.cache === 'object' ? this.config.cache : {});
    }

    // Initialize ANN Index if threshold reached or forced
    if (this.config.useANN || this.items.length >= this.config.annThreshold) {
      this.buildANNIndex();
    }

    // Initialize Updater
    this.updater = new BackgroundUpdater(this);
  }

  private buildANNIndex(): void {
    if (this.vectors.length === 0) return;
    const dims = this.vectors[0].length;
    const hnswConfig = typeof this.config.useANN === 'object' ? this.config.useANN : {};
    this.annIndex = new HNSWIndex(dims, hnswConfig);
    
    for (let i = 0; i < this.vectors.length; i++) {
      this.annIndex.add(i, this.vectors[i]);
    }
  }

  /**
   * Extract searchable text from an item using configured paths.
   */
  private getSearchableText(item: SearchItem<T>): string {
    return extractText(item, this.config.textPaths.length > 0 ? this.config.textPaths : undefined);
  }

  /**
   * Create a new Simile instance from items.
   * This will embed all items (slow for first run, but cached after).
   */
  static async from<T>(
    items: SearchItem<T>[],
    config: SimileConfig = {}
  ): Promise<Simile<T>> {
    const model = config.model ?? "Xenova/all-MiniLM-L6-v2";
    const textPaths = config.textPaths ?? [];
    
    // For initialization, we create a temporary cache to avoid duplicate embeddings
    // even if caching is disabled in config, it's useful during bulk init
    const tempCache = new VectorCache({ maxSize: items.length });
    const texts = items.map((item) => 
      extractText(item, textPaths.length > 0 ? textPaths : undefined)
    );
    
    const vectors: Float32Array[] = [];
    const textsToEmbed: string[] = [];
    const textToVectorIdx: Map<number, number> = new Map();

    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      const cacheKey = createCacheKey(text, model);
      const cached = tempCache.get(cacheKey);
      
      if (cached) {
        vectors[i] = cached;
      } else {
        textToVectorIdx.set(textsToEmbed.length, i);
        textsToEmbed.push(text);
      }
    }

    if (textsToEmbed.length > 0) {
      const newVectors = await embedBatch(textsToEmbed, model);
      for (let i = 0; i < newVectors.length; i++) {
        const originalIdx = textToVectorIdx.get(i)!;
        vectors[originalIdx] = newVectors[i];
        tempCache.set(createCacheKey(textsToEmbed[i], model), newVectors[i]);
      }
    }
    
    return new Simile<T>(items, vectors, config);
  }

  /**
   * Internal helper for embedding text with caching.
   */
  private async embedWithCache(text: string): Promise<Float32Array> {
    const cacheKey = createCacheKey(text, this.config.model);
    if (this.cache) {
      const cached = this.cache.get(cacheKey);
      if (cached) return cached;
    }

    const vector = await embed(text, this.config.model);
    
    if (this.cache) {
      this.cache.set(cacheKey, vector);
    }
    
    return vector;
  }

  /**
   * Load a Simile instance from a previously saved snapshot.
   * This is INSTANT - no embedding needed!
   */
  static load<T>(snapshot: SimileSnapshot<T>, config: SimileConfig = {}): Simile<T> {
    const vectors = snapshot.vectors.map(base64ToVector);
    return new Simile<T>(
      snapshot.items,
      vectors,
      { 
        ...config, 
        model: snapshot.model,
        textPaths: snapshot.textPaths ?? config.textPaths ?? [],
      }
    );
  }

  /**
   * Load from JSON string (e.g., from file or localStorage)
   */
  static loadFromJSON<T>(json: string, config: SimileConfig = {}): Simile<T> {
    const snapshot: SimileSnapshot<T> = JSON.parse(json);
    return Simile.load(snapshot, config);
  }

  /**
   * Save the current state to a snapshot object.
   * Store this in a file or database for instant loading later.
   */
  save(): SimileSnapshot<T> {
    return {
      version: PACKAGE_VERSION,
      model: this.config.model,
      items: this.items,
      vectors: this.vectors.map(vectorToBase64),
      createdAt: new Date().toISOString(),
      textPaths: this.config.textPaths.length > 0 ? this.config.textPaths : undefined,
    };
  }

  /**
   * Export as JSON string for file storage
   */
  toJSON(): string {
    return JSON.stringify(this.save());
  }

  async add(items: SearchItem<T>[]): Promise<void> {
    const texts = items.map((item) => this.getSearchableText(item));
    
    // Use embedBatch with cache optimization
    const newVectors: Float32Array[] = [];
    const textsToEmbed: string[] = [];
    const textToIdx: Map<number, number> = new Map();

    for (let i = 0; i < texts.length; i++) {
      const cacheKey = createCacheKey(texts[i], this.config.model);
      const cached = this.cache?.get(cacheKey);
      if (cached) {
        newVectors[i] = cached;
      } else {
        textToIdx.set(textsToEmbed.length, i);
        textsToEmbed.push(texts[i]);
      }
    }

    if (textsToEmbed.length > 0) {
      const embedded = await embedBatch(textsToEmbed, this.config.model);
      for (let i = 0; i < embedded.length; i++) {
        const originalIdx = textToIdx.get(i)!;
        newVectors[originalIdx] = embedded[i];
        this.cache?.set(createCacheKey(textsToEmbed[i], this.config.model), embedded[i]);
      }
    }

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const existingIdx = this.itemIndex.get(item.id);

      if (existingIdx !== undefined) {
        this.items[existingIdx] = item;
        this.vectors[existingIdx] = newVectors[i];
        this.annIndex?.remove(existingIdx);
        this.annIndex?.add(existingIdx, newVectors[i]);
      } else {
        const newIdx = this.items.length;
        this.items.push(item);
        this.vectors.push(newVectors[i]);
        this.itemIndex.set(item.id, newIdx);
        
        // Auto-enable ANN if threshold reached
        if (!this.annIndex && this.items.length >= this.config.annThreshold) {
          this.buildANNIndex();
        } else {
          this.annIndex?.add(newIdx, newVectors[i]);
        }
      }
    }
  }

  /**
   * Queue items for background indexing (non-blocking).
   */
  enqueue(items: SearchItem<T>[]): void {
    this.updater.enqueue(items);
  }

  /**
   * Get indexing information and stats.
   */
  getIndexInfo(): IndexInfo {
    let memoryBytes = 0;
    for (const v of this.vectors) memoryBytes += v.byteLength;
    
    return {
      type: this.annIndex ? 'hnsw' : 'linear',
      size: this.items.length,
      memory: `${(memoryBytes / 1024 / 1024).toFixed(2)} MB`,
      cacheStats: this.cache?.getStats(),
      annStats: this.annIndex?.getStats(),
    };
  }

  /**
   * Remove items by ID
   */
  remove(ids: string[]): void {
    const idsToRemove = new Set(ids);
    const newItems: SearchItem<T>[] = [];
    const newVectors: Float32Array[] = [];

    for (let i = 0; i < this.items.length; i++) {
      if (!idsToRemove.has(this.items[i].id)) {
        newItems.push(this.items[i]);
        newVectors.push(this.vectors[i]);
      }
    }

    this.items = newItems;
    this.vectors = newVectors;
    this.itemIndex = new Map(this.items.map((item, i) => [item.id, i]));
    
    // Rebuild ANN index if it exists
    if (this.annIndex) {
      this.buildANNIndex();
    }
  }

  /**
   * Get item by ID
   */
  get(id: string): SearchItem<T> | undefined {
    const idx = this.itemIndex.get(id);
    return idx !== undefined ? this.items[idx] : undefined;
  }

  /**
   * Get all items
   */
  getAll(): SearchItem<T>[] {
    return [...this.items];
  }

  /**
   * Get the number of items in the index
   */
  get size(): number {
    return this.items.length;
  }

  /**
   * Set custom scoring weights
   */
  setWeights(weights: HybridWeights): void {
    this.config.weights = { ...this.config.weights, ...weights };
  }

  /**
   * Search for similar items.
   * 
   * @param query - The search query
   * @param options - Search options
   * @returns Sorted results by relevance (highest score first)
   */
  async search(
    query: string,
    options: SearchOptions = {}
  ): Promise<SearchResult<T>[]> {
    const {
      topK = 5,
      explain = false,
      filter,
      threshold = 0,
      minLength = 1,
    } = options;

    // Min character limit - don't search until query meets minimum length
    if (query.length < minLength) {
      return [];
    }

    const qVector = await this.embedWithCache(query);

    // First pass: calculate raw scores
    const rawResults: Array<{
      index: number;
      item: SearchItem<T>;
      semantic: number;
      fuzzy: number;
      keyword: number;
    }> = [];

    // Use ANN if enabled and available
    if (this.annIndex && (options.useANN ?? true)) {
      const annResults = this.annIndex.search(qVector, topK * 2); // Get more for filtering
      for (const res of annResults) {
        const item = this.items[res.id];
        if (filter && !filter(item.metadata)) continue;

        const searchableText = this.getSearchableText(item);
        const semantic = 1 - res.distance; // distance to similarity
        const fuzzy = fuzzyScore(query, searchableText);
        const keyword = keywordScore(query, searchableText);

        rawResults.push({ index: res.id, item, semantic, fuzzy, keyword });
      }
    } else {
      // Fallback to linear scan
      for (let i = 0; i < this.items.length; i++) {
        const item = this.items[i];

        if (filter && !filter(item.metadata)) continue;

        const searchableText = this.getSearchableText(item);
        const semantic = cosine(qVector, this.vectors[i]);
        const fuzzy = fuzzyScore(query, searchableText);
        const keyword = keywordScore(query, searchableText);

        rawResults.push({ index: i, item, semantic, fuzzy, keyword });
      }
    }

    // Calculate score statistics for normalization
    const stats = calculateScoreStats(rawResults);

    // Second pass: normalize scores and compute hybrid score
    const results: SearchResult<T>[] = [];

    for (const raw of rawResults) {
      let semantic = raw.semantic;
      let fuzzy = raw.fuzzy;
      let keyword = raw.keyword;

      // Normalize scores if enabled
      if (this.config.normalizeScores) {
        semantic = normalizeScore(raw.semantic, stats.semantic.min, stats.semantic.max);
        fuzzy = normalizeScore(raw.fuzzy, stats.fuzzy.min, stats.fuzzy.max);
        keyword = normalizeScore(raw.keyword, stats.keyword.min, stats.keyword.max);
      }

      const score = hybridScore(semantic, fuzzy, keyword, this.config.weights);

      // Apply threshold filter
      if (score < threshold) continue;

      results.push({
        id: raw.item.id,
        text: raw.item.text,
        metadata: raw.item.metadata,
        score,
        explain: explain
          ? {
              semantic,
              fuzzy,
              keyword,
              raw: {
                semantic: raw.semantic,
                fuzzy: raw.fuzzy,
                keyword: raw.keyword,
              },
            }
          : undefined,
      });
    }

    // Sort by relevance (highest score first)
    return results.sort((a, b) => b.score - a.score).slice(0, topK);
  }
}
