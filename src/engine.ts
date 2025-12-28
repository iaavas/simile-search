import { embed, embedBatch, vectorToBase64, base64ToVector } from "./embedder.js";
import { cosine, fuzzyScore, keywordScore, calculateScoreStats } from "./similarity.js";
import { hybridScore, getDefaultWeights } from "./ranker.js";
import { extractText, normalizeScore } from "./utils.js";
import { SearchItem, SearchResult, SearchOptions, SimileConfig, SimileSnapshot, HybridWeights } from "./types.js";


const PACKAGE_VERSION = "0.3.2";

export class Simile<T = any> {
  private items: SearchItem<T>[];
  private vectors: Float32Array[];
  private itemIndex: Map<string, number>;
  private config: Required<SimileConfig>;

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
    };
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
    
    // Extract text using paths if configured
    const texts = items.map((item) => 
      extractText(item, textPaths.length > 0 ? textPaths : undefined)
    );
    
    const vectors = await embedBatch(texts, model);
    return new Simile<T>(items, vectors, config);
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

  /**
   * Add new items to the index
   */
  async add(items: SearchItem<T>[]): Promise<void> {
    const texts = items.map((item) => this.getSearchableText(item));
    const newVectors = await embedBatch(texts, this.config.model);

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const existingIdx = this.itemIndex.get(item.id);

      if (existingIdx !== undefined) {
        // Update existing item
        this.items[existingIdx] = item;
        this.vectors[existingIdx] = newVectors[i];
      } else {
        // Add new item
        const newIdx = this.items.length;
        this.items.push(item);
        this.vectors.push(newVectors[i]);
        this.itemIndex.set(item.id, newIdx);
      }
    }
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

    const qVector = await embed(query, this.config.model);

    // First pass: calculate raw scores
    const rawResults: Array<{
      index: number;
      item: SearchItem<T>;
      semantic: number;
      fuzzy: number;
      keyword: number;
    }> = [];

    for (let i = 0; i < this.items.length; i++) {
      const item = this.items[i];

      if (filter && !filter(item.metadata)) continue;

      const searchableText = this.getSearchableText(item);
      const semantic = cosine(qVector, this.vectors[i]);
      const fuzzy = fuzzyScore(query, searchableText);
      const keyword = keywordScore(query, searchableText);

      rawResults.push({ index: i, item, semantic, fuzzy, keyword });
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
