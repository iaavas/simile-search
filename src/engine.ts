import { embed, embedBatch, vectorToBase64, base64ToVector } from "./embedder";
import { cosine, fuzzyScore, keywordScore } from "./similarity";
import { hybridScore, getDefaultWeights } from "./ranker";
import {
  SearchItem,
  SearchResult,
  SearchOptions,
  SimileConfig,
  SimileSnapshot,
  HybridWeights,
} from "./types";

const PACKAGE_VERSION = "0.2.0";

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
    };
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
    const texts = items.map((item) => item.text);
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
      { ...config, model: snapshot.model }
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
    const texts = items.map((item) => item.text);
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
   * Search for similar items
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
    } = options;

    const qVector = await embed(query, this.config.model);

    const results: SearchResult<T>[] = [];

    for (let i = 0; i < this.items.length; i++) {
      const item = this.items[i];

      if (filter && !filter(item.metadata)) continue;

      const semantic = cosine(qVector, this.vectors[i]);
      const fuzzy = fuzzyScore(query, item.text);
      const keyword = keywordScore(query, item.text);

      const score = hybridScore(semantic, fuzzy, keyword, this.config.weights);

      // Apply threshold filter
      if (score < threshold) continue;

      results.push({
        id: item.id,
        text: item.text,
        metadata: item.metadata,
        score,
        explain: explain ? { semantic, fuzzy, keyword } : undefined,
      });
    }

    return results.sort((a, b) => b.score - a.score).slice(0, topK);
  }
}
