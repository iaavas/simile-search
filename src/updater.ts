/**
 * Background Updater - Non-blocking updates with queue-based processing.
 * 
 * Allows adding items asynchronously without blocking search operations.
 * Items are batched and processed in the background for efficiency.
 */

import { SearchItem } from "./types.js";

export interface UpdaterConfig {
  /** Milliseconds to wait before processing queued items (default: 100) */
  batchDelay?: number;
  /** Maximum items to process in a single batch (default: 50) */
  maxBatchSize?: number;
  /** Progress callback */
  onProgress?: (processed: number, total: number) => void;
  /** Error callback */
  onError?: (error: Error, item: SearchItem) => void;
}

export interface UpdaterStats {
  /** Total items processed since creation */
  totalProcessed: number;
  /** Current queue size */
  pendingCount: number;
  /** Whether currently processing */
  isProcessing: boolean;
  /** Average items per batch */
  avgBatchSize: number;
  /** Total batches processed */
  batchCount: number;
}

type UpdateEventType = 'complete' | 'error' | 'batch' | 'progress';
type UpdateEventHandler = (...args: any[]) => void;

/**
 * Interface for the Simile engine (to avoid circular dependencies).
 */
interface SimileInterface<T> {
  add(items: SearchItem<T>[]): Promise<void>;
}

/**
 * Background updater for non-blocking item additions.
 */
export class BackgroundUpdater<T = any> {
  private simile: SimileInterface<T>;
  private config: Required<Omit<UpdaterConfig, 'onProgress' | 'onError'>> & {
    onProgress?: UpdaterConfig['onProgress'];
    onError?: UpdaterConfig['onError'];
  };
  private _queue: SearchItem<T>[] = [];
  private processing: boolean = false;
  private timeoutId: ReturnType<typeof setTimeout> | null = null;
  private eventHandlers: Map<UpdateEventType, Set<UpdateEventHandler>> = new Map();
  
  // Stats
  private totalProcessed: number = 0;
  private batchCount: number = 0;
  private totalBatchItems: number = 0;

  constructor(simile: SimileInterface<T>, config: UpdaterConfig = {}) {
    this.simile = simile;
    this.config = {
      batchDelay: config.batchDelay ?? 100,
      maxBatchSize: config.maxBatchSize ?? 50,
      onProgress: config.onProgress,
      onError: config.onError,
    };
  }

  /**
   * Queue items for background embedding.
   * Items are batched and processed after batchDelay ms.
   */
  enqueue(items: SearchItem<T>[]): void {
    this._queue.push(...items);
    this.scheduleProcessing();
  }

  /**
   * Queue a single item.
   */
  enqueueOne(item: SearchItem<T>): void {
    this._queue.push(item);
    this.scheduleProcessing();
  }

  /**
   * Force immediate processing of queued items.
   */
  async flush(): Promise<void> {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    await this.processQueue();
  }

  /**
   * Wait for all pending items to be processed.
   */
  async waitForCompletion(): Promise<void> {
    while (this._queue.length > 0 || this.processing) {
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }

  /**
   * Get the number of items waiting to be processed.
   */
  getPendingCount(): number {
    return this._queue.length;
  }

  /**
   * Check if currently processing.
   */
  isProcessing(): boolean {
    return this.processing;
  }

  /**
   * Clear all pending items without processing.
   */
  clear(): void {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    this._queue = [];
  }

  /**
   * Get updater statistics.
   */
  getStats(): UpdaterStats {
    return {
      totalProcessed: this.totalProcessed,
      pendingCount: this._queue.length,
      isProcessing: this.processing,
      avgBatchSize: this.batchCount > 0 ? this.totalBatchItems / this.batchCount : 0,
      batchCount: this.batchCount,
    };
  }

  /**
   * Reset statistics.
   */
  resetStats(): void {
    this.totalProcessed = 0;
    this.batchCount = 0;
    this.totalBatchItems = 0;
  }

  /**
   * Register an event handler.
   */
  on(event: UpdateEventType, handler: UpdateEventHandler): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler);
  }

  /**
   * Remove an event handler.
   */
  off(event: UpdateEventType, handler: UpdateEventHandler): void {
    this.eventHandlers.get(event)?.delete(handler);
  }

  /**
   * Emit an event to all registered handlers.
   */
  private emit(event: UpdateEventType, ...args: any[]): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      for (const handler of handlers) {
        try {
          handler(...args);
        } catch (e) {
          console.error(`Error in ${event} handler:`, e);
        }
      }
    }
  }

  private scheduleProcessing(): void {
    if (this.timeoutId || this.processing) return;
    
    this.timeoutId = setTimeout(() => {
      this.timeoutId = null;
      this.processQueue().catch(console.error);
    }, this.config.batchDelay);
  }

  private async processQueue(): Promise<void> {
    if (this.processing || this._queue.length === 0) return;

    this.processing = true;
    const total = this._queue.length;

    try {
      while (this._queue.length > 0) {
        // Take batch from queue
        const batch = this._queue.splice(0, this.config.maxBatchSize);
        const processed = total - this._queue.length;
        
        this.emit('batch', batch, processed, total);

        try {
          await this.simile.add(batch);
          
          this.totalProcessed += batch.length;
          this.batchCount++;
          this.totalBatchItems += batch.length;
          
          if (this.config.onProgress) {
            this.config.onProgress(processed, total);
          }
          this.emit('progress', processed, total);
        } catch (error) {
          // Handle errors per-item if possible
          for (const item of batch) {
            if (this.config.onError) {
              this.config.onError(error as Error, item);
            }
            this.emit('error', error, item);
          }
        }
      }

      this.emit('complete', this.totalProcessed);
    } finally {
      this.processing = false;
    }
  }
}

/**
 * Debounced updater - coalesces rapid updates.
 * Useful when items change frequently (e.g., user typing).
 */
export class DebouncedUpdater<T = any> {
  private updater: BackgroundUpdater<T>;
  private debounceMs: number;
  private pending: Map<string, SearchItem<T>> = new Map();
  private timeoutId: ReturnType<typeof setTimeout> | null = null;

  constructor(simile: SimileInterface<T>, debounceMs: number = 300, config: UpdaterConfig = {}) {
    this.updater = new BackgroundUpdater(simile, config);
    this.debounceMs = debounceMs;
  }

  /**
   * Queue an item for update. If same ID is queued again before flush,
   * only the latest version is processed.
   */
  update(item: SearchItem<T>): void {
    this.pending.set(item.id, item);
    this.scheduleFlush();
  }

  /**
   * Force immediate flush of pending items.
   */
  async flush(): Promise<void> {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    await this.doFlush();
  }

  private scheduleFlush(): void {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
    }
    this.timeoutId = setTimeout(() => {
      this.timeoutId = null;
      this.doFlush().catch(console.error);
    }, this.debounceMs);
  }

  private async doFlush(): Promise<void> {
    const items = Array.from(this.pending.values());
    this.pending.clear();
    
    if (items.length > 0) {
      this.updater.enqueue(items);
      await this.updater.flush();
    }
  }

  /**
   * Get pending count (not yet flushed + in queue).
   */
  getPendingCount(): number {
    return this.pending.size + this.updater.getPendingCount();
  }

  /**
   * Clear all pending updates.
   */
  clear(): void {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    this.pending.clear();
    this.updater.clear();
  }

  /**
   * Get the underlying updater for event handling.
   */
  getUpdater(): BackgroundUpdater<T> {
    return this.updater;
  }
}

/**
 * Priority queue for updates - high priority items processed first.
 */
export class PriorityUpdater<T = any> {
  private simile: SimileInterface<T>;
  private config: UpdaterConfig;
  private highPriority: SearchItem<T>[] = [];
  private normalPriority: SearchItem<T>[] = [];
  private processing: boolean = false;

  constructor(simile: SimileInterface<T>, config: UpdaterConfig = {}) {
    this.simile = simile;
    this.config = {
      batchDelay: config.batchDelay ?? 50,
      maxBatchSize: config.maxBatchSize ?? 50,
      ...config,
    };
  }

  /**
   * Queue high priority item (processed first).
   */
  queueHigh(items: SearchItem<T>[]): void {
    this.highPriority.push(...items);
    this.scheduleProcessing();
  }

  /**
   * Queue normal priority item.
   */
  enqueue(items: SearchItem<T>[]): void {
    this.normalPriority.push(...items);
    this.scheduleProcessing();
  }

  private timeoutId: ReturnType<typeof setTimeout> | null = null;

  private scheduleProcessing(): void {
    if (this.timeoutId || this.processing) return;
    
    this.timeoutId = setTimeout(() => {
      this.timeoutId = null;
      this.processQueue().catch(console.error);
    }, this.config.batchDelay);
  }

  private async processQueue(): Promise<void> {
    if (this.processing) return;
    
    this.processing = true;
    const maxBatch = this.config.maxBatchSize ?? 50;

    try {
      // Process high priority first
      while (this.highPriority.length > 0) {
        const batch = this.highPriority.splice(0, maxBatch);
        await this.simile.add(batch);
      }

      // Then normal priority
      while (this.normalPriority.length > 0) {
        const batch = this.normalPriority.splice(0, maxBatch);
        await this.simile.add(batch);
      }
    } finally {
      this.processing = false;
    }
  }

  /**
   * Force immediate processing.
   */
  async flush(): Promise<void> {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    await this.processQueue();
  }

  /**
   * Get total pending count.
   */
  getPendingCount(): number {
    return this.highPriority.length + this.normalPriority.length;
  }

  /**
   * Clear all pending items.
   */
  clear(): void {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    this.highPriority = [];
    this.normalPriority = [];
  }
}
