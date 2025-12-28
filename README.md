<div align="center">
  <img src="assets/logo.svg" alt="Simile Logo" width="200">
  
  # Simile
  
  **Intelligent offline-first semantic search for modern applications**
  
  [![npm version](https://img.shields.io/npm/v/simile-search)](https://www.npmjs.com/package/simile-search)
  [![npm downloads](https://img.shields.io/npm/dm/simile-search)](https://www.npmjs.com/package/simile-search)
  [![license](https://img.shields.io/npm/l/simile-search)](https://github.com/iaavas/simile/blob/main/LICENSE)
  
</div>

---

## Overview

Simile is a high-performance search engine that combines semantic understanding, fuzzy matching, and keyword search to deliver highly relevant results—entirely offline. Built with Transformers.js, it requires no API calls, runs completely locally, and scales to handle large datasets efficiently.

Perfect for product catalogs, content libraries, user directories, and any application requiring intelligent search without external dependencies.

## Key Features

- **🧠 Semantic Understanding** — Finds conceptually similar items, not just keyword matches ("phone charger" → "USB-C cable")
- **🔤 Typo Tolerance** — Fuzzy matching handles misspellings and partial queries gracefully
- **⚡ Lightning Fast** — O(log n) search with HNSW indexing for datasets of 10k+ items
- **💾 Memory Efficient** — Quantization support (float16/int8) reduces memory usage by up to 75%
- **🔄 Non-blocking Updates** — Asynchronous indexing keeps your application responsive
- **📦 Zero Dependencies on APIs** — Runs entirely locally with Transformers.js
- **🔗 Deep Object Search** — Query nested fields with dot notation (`author.firstName`)
- **💾 Persistent Storage** — Save and load embeddings to avoid recomputation
- **🎯 Highly Configurable** — Tune scoring weights, thresholds, and search behavior

## Installation

```bash
npm install simile-search
```

## Quick Start

```typescript
import { Simile } from 'simile-search';

// Initialize search engine
const engine = await Simile.from([
  { id: '1', text: 'Bathroom floor cleaner', metadata: { category: 'Cleaning' } },
  { id: '2', text: 'Dishwashing liquid', metadata: { category: 'Kitchen' } },
  { id: '3', text: 'iPhone Charger', metadata: { category: 'Electronics' } },
  { id: '4', text: 'USB-C phone charger cable', metadata: { category: 'Electronics' } },
]);

// Search with natural language
const results = await engine.search('phone charger');
console.log(results);
// [
//   { id: '3', text: 'iPhone Charger', score: 0.92, ... },
//   { id: '4', text: 'USB-C phone charger cable', score: 0.87, ... }
// ]
```

## Core Concepts

### Persistence

Avoid re-embedding on every startup by saving your index:

```typescript
import { Simile } from 'simile-search';
import * as fs from 'fs';

// Initial setup: embed and save
const engine = await Simile.from(items);
fs.writeFileSync('search-index.json', engine.toJSON());

// Subsequent loads: instant startup
const json = fs.readFileSync('search-index.json', 'utf-8');
const loadedEngine = Simile.loadFromJSON(json);

// Functionally identical to the original
const results = await loadedEngine.search('query');
```

**Snapshot Format** for database storage:

```typescript
const snapshot = engine.save();
// {
//   version: '0.2.0',
//   model: 'Xenova/all-MiniLM-L6-v2',
//   items: [...],
//   vectors: ['base64...'],
//   createdAt: '2024-12-28T...',
//   textPaths: [...]
// }

const restored = Simile.load(snapshot);
```

### Nested Object Search

Search complex data structures by specifying extraction paths:

```typescript
const books = [
  {
    id: '1',
    metadata: {
      author: { firstName: 'John', lastName: 'Doe' },
      title: 'The Art of Programming',
      tags: ['coding', 'javascript'],
    },
  },
];

const engine = await Simile.from(books, {
  textPaths: [
    'metadata.author.firstName',
    'metadata.author.lastName',
    'metadata.title',
    'metadata.tags',  // Arrays are automatically joined
  ],
});

// Search across all configured paths
const results = await engine.search('John programming');
```

**Supported path formats:**
- Nested objects: `metadata.author.firstName`
- Array indexing: `items[0].name`
- Array joining: `metadata.tags` (joins all elements)

### Dynamic Catalog Management

Update your search index without rebuilding:

```typescript
// Add new items
await engine.add([
  { id: '5', text: 'Wireless headphones', metadata: { category: 'Electronics' } }
]);

// Update existing items (by ID)
await engine.add([
  { id: '1', text: 'Premium bathroom cleaner', metadata: { category: 'Cleaning' } }
]);

// Remove items
engine.remove(['2', '3']);

// Retrieve items
const item = engine.get('1');
const allItems = engine.getAll();
console.log(engine.size); // Current item count
```

## Configuration

### Scoring Weights

Customize how different matching strategies contribute to the final score:

```typescript
const engine = await Simile.from(items, {
  weights: {
    semantic: 0.7,  // AI embedding similarity (default)
    fuzzy: 0.15,    // Levenshtein distance
    keyword: 0.15,  // Exact keyword matching
  }
});

// Adjust weights dynamically
engine.setWeights({ semantic: 0.9, fuzzy: 0.05, keyword: 0.05 });
```

### Score Normalization

Simile normalizes scores across different matching methods for fair comparison:

```typescript
const engine = await Simile.from(items, {
  normalizeScores: true,  // Enabled by default
});

// View normalized and raw scores
const results = await engine.search('cleaner', { explain: true });
// {
//   score: 1.0,
//   explain: {
//     semantic: 1.0,    // normalized
//     fuzzy: 1.0,       // normalized
//     keyword: 1.0,     // normalized
//     raw: {
//       semantic: 0.62,
//       fuzzy: 0.32,
//       keyword: 1.0
//     }
//   }
// }
```

### Search Options

Fine-tune search behavior per query:

```typescript
const results = await engine.search('cleaner', {
  topK: 10,                                      // Maximum results (default: 5)
  threshold: 0.5,                                // Minimum score cutoff
  explain: true,                                 // Include score breakdown
  filter: (meta) => meta.category === 'Cleaning', // Metadata filtering
  minLength: 3,                                  // Minimum query length (default: 1)
});
```

**Minimum character limit** prevents unnecessary searches on partial input:

```typescript
await engine.search('cl', { minLength: 3 }); // Returns [] (too short)
await engine.search('cle', { minLength: 3 }); // Returns results
```

## Performance Optimization

Simile is designed to scale efficiently from hundreds to hundreds of thousands of items.

### Quantization

Reduce memory usage with lower-precision vector representations:

```typescript
const engine = await Simile.from(items, {
  quantization: 'float16', // 50% memory reduction, minimal accuracy loss
  // OR
  quantization: 'int8',    // 75% memory reduction, slight accuracy trade-off
});
```

### Approximate Nearest Neighbor (ANN) Search

For large datasets, HNSW indexing provides logarithmic search time:

```typescript
const engine = await Simile.from(items, {
  useANN: true,          // Enable ANN indexing
  annThreshold: 1000,    // Auto-enable when items > threshold (default: 1000)
});
```

### Vector Caching

LRU cache eliminates redundant embeddings for duplicate texts:

```typescript
const engine = await Simile.from(items, {
  cache: {
    maxSize: 5000,      // Cache up to 5000 embeddings
    enableStats: true,  // Track cache performance
  }
});

// Monitor cache efficiency
const stats = engine.getIndexInfo().cacheStats;
console.log(`Hit rate: ${stats.hitRate}%`);
```

### Background Indexing

Updates are processed asynchronously to maintain responsiveness:

```typescript
// Returns immediately, processes in background
await engine.add(newItems);
await engine.add(moreItems);
```

## Advanced Usage

### Direct Utility Access

For custom implementations:

```typescript
import { 
  embed, 
  embedBatch, 
  cosine, 
  fuzzyScore, 
  keywordScore,
  hybridScore,
  getByPath,
  extractText,
} from 'simile-search';

// Generate embeddings
const vector = await embed('hello world');
const vectors = await embedBatch(['text1', 'text2', 'text3']);

// Calculate similarities
const similarity = cosine(vectorA, vectorB);
const fuzzy = fuzzyScore('cleaner', 'cleenr');
const keyword = keywordScore('phone charger', 'USB phone charger cable');

// Combine scores
const finalScore = hybridScore(
  0.8, 0.6, 0.5,
  { semantic: 0.7, fuzzy: 0.15, keyword: 0.15 }
);

// Extract nested data
const firstName = getByPath(obj, 'author.firstName');
const text = extractText(item, ['metadata.title', 'metadata.tags']);
```

## API Reference

### Class Methods

| Method | Description |
|--------|-------------|
| `Simile.from(items, config?)` | Create engine from items (async, embeds all) |
| `Simile.load(snapshot, config?)` | Load from snapshot object (instant) |
| `Simile.loadFromJSON(json, config?)` | Load from JSON string |
| `engine.search(query, options?)` | Search for similar items (sorted by relevance) |
| `engine.save()` | Export snapshot object |
| `engine.toJSON()` | Export as JSON string |
| `engine.add(items)` | Add or update items (async) |
| `engine.remove(ids)` | Remove items by ID |
| `engine.get(id)` | Retrieve single item |
| `engine.getAll()` | Retrieve all items |
| `engine.setWeights(weights)` | Update scoring weights |
| `engine.size` | Current item count |

## TypeScript Types

```typescript
interface SearchItem<T = any> {
  id: string;
  text: string;
  metadata?: T;
}

interface SearchResult<T = any> {
  id: string;
  text: string;
  score: number;
  metadata?: T;
  explain?: {
    semantic: number;
    fuzzy: number;
    keyword: number;
    raw?: { semantic: number; fuzzy: number; keyword: number };
  };
}

interface SearchOptions {
  topK?: number;
  explain?: boolean;
  threshold?: number;
  minLength?: number;
  filter?: (metadata: any) => boolean;
}

interface SimileConfig {
  weights?: { semantic?: number; fuzzy?: number; keyword?: number };
  model?: string;
  textPaths?: string[];
  normalizeScores?: boolean;
  cache?: boolean | CacheOptions;
  quantization?: 'float32' | 'float16' | 'int8';
  useANN?: boolean | HNSWConfig;
  annThreshold?: number;
}
```

## Technical Details

**Embedding Model:** [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) via Transformers.js

This model runs entirely in JavaScript with no Python runtime or external API dependencies.

## License

MIT © [Aavash Baral](https://github.com/iaavas)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/iaavas">Aavash Baral</a></sub>
</div>