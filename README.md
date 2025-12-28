<div align="center">
  <img src="assets/logo.jpeg" alt="Simile Logo" width="200">
</div>

# Simile 🔍

![npm](https://img.shields.io/npm/v/simile-search)
![npm](https://img.shields.io/npm/dm/simile-search)
![license](https://img.shields.io/npm/l/simile-search)

**Offline-first semantic + fuzzy search engine for catalogs, names, and products.**

Simile combines the power of AI embeddings with fuzzy string matching and keyword search to deliver highly relevant search results—all running locally, no API calls required.

## ✨ Features

- 🧠 **Semantic Search** - Understands meaning, not just keywords ("phone charger" finds "USB-C cable")
- 🔤 **Fuzzy Matching** - Handles typos and partial matches gracefully
- 🎯 **Keyword Boost** - Exact matches get priority
- ⚡ **O(log n) Search** - Built-in HNSW index for lightning-fast search on large datasets (10k+ items)
- 📉 **Quantization** - Reduce memory usage by up to 75% with `float16` and `int8` support
- 🚀 **Vector Cache** - LRU caching to avoid redundant embedding of duplicate text
- 🔄 **Non-blocking Updates** - Asynchronous background indexing keeps your app responsive
- 💾 **Persistence** - Save/load embeddings to avoid re-computing
- 🔧 **Configurable** - Tune scoring weights for your use case
- 📦 **Zero API Calls** - Everything runs locally with Transformers.js
- 🔗 **Nested Path Search** - Search `author.firstName` instead of flat strings
- 📊 **Score Normalization** - Consistent scoring across different methods
- ✂️ **Min Character Limit** - Control when search triggers

## 📦 Installation

```bash
npm install simile-search
```

## 🚀 Quick Start

```typescript
import { Simile } from 'simile-search';

// Create a search engine with your items
const engine = await Simile.from([
  { id: '1', text: 'Bathroom floor cleaner', metadata: { category: 'Cleaning' } },
  { id: '2', text: 'Dishwashing liquid', metadata: { category: 'Kitchen' } },
  { id: '3', text: 'iPhone Charger', metadata: { category: 'Electronics' } },
  { id: '4', text: 'USB-C phone charger cable', metadata: { category: 'Electronics' } },
]);

// Search!
const results = await engine.search('phone charger');
console.log(results);
// [
//   { id: '3', text: 'iPhone Charger', score: 0.92, ... },
//   { id: '4', text: 'USB-C phone charger cable', score: 0.87, ... },
//   ...
// ]
```

## 💾 Persistence (Save & Load)

The first embedding run can be slow. Save your embeddings to load instantly next time:

```typescript
import { Simile } from 'simile-search';
import * as fs from 'fs';

// First run: embed and save (slow, but only once!)
const engine = await Simile.from(items);
fs.writeFileSync('catalog.json', engine.toJSON());

// Later: instant load from file (no re-embedding!)
const json = fs.readFileSync('catalog.json', 'utf-8');
const loadedEngine = Simile.loadFromJSON(json);

// Works exactly the same
const results = await loadedEngine.search('cleaner');
```

### Snapshot Format

```typescript
// For database storage
const snapshot = engine.save();
// {
//   version: '0.2.0',
//   model: 'Xenova/all-MiniLM-L6-v2',
//   items: [...],
//   vectors: ['base64...', 'base64...'],
//   createdAt: '2024-12-28T...',
//   textPaths: ['metadata.title', ...]  // if configured
// }

// Load from snapshot object
const restored = Simile.load(snapshot);
```

## 🔗 Nested Path Search

Search complex objects by specifying paths to extract text from:

```typescript
const books = [
  {
    id: '1',
    text: '',  // Can be empty when using textPaths
    metadata: {
      author: { firstName: 'John', lastName: 'Doe' },
      title: 'The Art of Programming',
      tags: ['coding', 'javascript'],
    },
  },
  {
    id: '2',
    text: '',
    metadata: {
      author: { firstName: 'Jane', lastName: 'Smith' },
      title: 'Machine Learning Basics',
      tags: ['ai', 'python'],
    },
  },
];

// Configure which paths to extract and search
const engine = await Simile.from(books, {
  textPaths: [
    'metadata.author.firstName',
    'metadata.author.lastName',
    'metadata.title',
    'metadata.tags',  // Arrays are joined with spaces
  ],
});

// Now you can search by author name!
const results = await engine.search('John programming');
// Finds "The Art of Programming" by John Doe
```

### Supported Path Formats

```typescript
// Dot notation for nested objects
'metadata.author.firstName'  // → "John"

// Array index access
'metadata.tags[0]'           // → "coding"
'items[0].name'              // → nested array access

// Arrays without index (joins all elements)
'metadata.tags'              // → "coding javascript"
```

## 🔧 Configuration

### Custom Scoring Weights

Tune how much each scoring method contributes:

```typescript
const engine = await Simile.from(items, {
  weights: {
    semantic: 0.7,  // AI embedding similarity (default: 0.7)
    fuzzy: 0.15,    // Levenshtein distance (default: 0.15)
    keyword: 0.15,  // Exact keyword matches (default: 0.15)
  }
});

// Or adjust later
engine.setWeights({ semantic: 0.9, fuzzy: 0.05, keyword: 0.05 });
```

### Score Normalization

By default, scores are normalized so that a "0.8" semantic score means the same as a "0.8" fuzzy score. This ensures fair comparison across different scoring methods.

```typescript
// Enabled by default
const engine = await Simile.from(items, {
  normalizeScores: true,  // default
});

// Disable if you want raw scores
const rawEngine = await Simile.from(items, {
  normalizeScores: false,
});

// With explain: true, you can see both normalized and raw scores
const results = await engine.search('cleaner', { explain: true });
// {
//   score: 1.0,
//   explain: {
//     semantic: 1.0,    // normalized
//     fuzzy: 1.0,       // normalized
//     keyword: 1.0,     // normalized
//     raw: {
//       semantic: 0.62, // original score
//       fuzzy: 0.32,    // original score
//       keyword: 1.0,   // original score
//     }
//   }
// }
```

### Search Options

```typescript
const results = await engine.search('cleaner', {
  topK: 10,           // Max results to return (default: 5)
  threshold: 0.5,     // Minimum score (default: 0)
  explain: true,      // Include score breakdown
  filter: (meta) => meta.category === 'Cleaning',  // Filter by metadata
  minLength: 3,       // Don't search until 3+ characters typed (default: 1)
});
```

### Min Character Limit

Prevent unnecessary searches on very short queries:

```typescript
// Don't trigger search until user types at least 3 characters
const results = await engine.search('cl', { minLength: 3 });
// Returns [] because query length (2) < minLength (3)

const results2 = await engine.search('cle', { minLength: 3 });
// Returns results because query length (3) >= minLength (3)
```

This is useful for autocomplete/typeahead UIs where you don't want to search on every keystroke.

## 📝 Dynamic Catalog Management

Add, update, or remove items without rebuilding:

```typescript
// Add new items
await engine.add([
  { id: '5', text: 'Wireless headphones', metadata: { category: 'Electronics' } }
]);

// Update existing item (same ID)
await engine.add([
  { id: '1', text: 'Premium bathroom cleaner', metadata: { category: 'Cleaning' } }
]);

// Remove items
engine.remove(['2', '3']);

// Get item by ID
const item = engine.get('1');

// Get all items
const allItems = engine.getAll();

// Get count
console.log(engine.size); // 3
```

## 🎯 Advanced: Direct Access to Utilities

For custom implementations:

```typescript
import { 
  embed, 
  embedBatch, 
  cosine, 
  fuzzyScore, 
  keywordScore,
  hybridScore,
  vectorToBase64,
  base64ToVector,
  getByPath,
  extractText,
  normalizeScore,
  calculateScoreStats,
} from 'simile-search';

// Embed text directly
const vector = await embed('hello world');

// Batch embed for performance
const vectors = await embedBatch(['text1', 'text2', 'text3']);

// Calculate similarities
const similarity = cosine(vectorA, vectorB);
const fuzzy = fuzzyScore('cleaner', 'cleenr');
const keyword = keywordScore('phone charger', 'USB phone charger cable');

// Combine scores
const score = hybridScore(0.8, 0.6, 0.5, { semantic: 0.7, fuzzy: 0.15, keyword: 0.15 });

// Extract nested values
const firstName = getByPath(obj, 'author.firstName');
const text = extractText(item, ['metadata.title', 'metadata.tags']);
```

## 📊 API Reference

### `Simile.from(items, config?)`
Create a new engine from items. Embeds all items (async).

### `Simile.load(snapshot, config?)`
Load from a saved snapshot (instant, no embedding).

### `Simile.loadFromJSON(json, config?)`
Load from JSON string.

### `engine.search(query, options?)`
Search for similar items. **Results are always sorted by relevance (highest score first).**

### `engine.save()`
Export snapshot object for persistence.

### `engine.toJSON()`
Export as JSON string.

### `engine.add(items)`
Add or update items (async).

### `engine.remove(ids)`
Remove items by ID.

### `engine.get(id)`
Get single item by ID.

### `engine.getAll()`
Get all items.

### `engine.size`
Number of items.

### `engine.setWeights(weights)`
Update scoring weights.

## 🧪 Types

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
  minLength?: number;  // Min query length to trigger search
  filter?: (metadata: any) => boolean;
}

interface SimileConfig {
  weights?: { semantic?: number; fuzzy?: number; keyword?: number };
  model?: string;
  textPaths?: string[];       // Paths for nested object search
  normalizeScores?: boolean;  // Enable score normalization (default: true)
  cache?: boolean | CacheOptions;
  quantization?: 'float32' | 'float16' | 'int8';
  useANN?: boolean | HNSWConfig;
  annThreshold?: number;
}

interface CacheOptions {
  maxSize?: number;
  enableStats?: boolean;
}

interface HNSWConfig {
  M?: number;
  efConstruction?: number;
  efSearch?: number;
}
```

## 🤖 Model

Simile uses [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) via Transformers.js by default. This model runs entirely in JavaScript—no Python or external APIs required.

## 📄 License

MIT © [Aavash Baral](https://github.com/iaavas)

## ⚡ Performance Optimization

Simile v0.4.0 introduces several features to handle large scale datasets (10k-100k+ items) efficiently.

### 📉 Quantization

Reduce memory footprint by representing vectors with lower precision.

```typescript
const engine = await Simile.from(items, {
  quantization: 'float16', // 50% memory reduction, minimal accuracy loss
  // OR
  quantization: 'int8',    // 75% memory reduction, slight accuracy loss
});
```

### ⚡ O(log n) Search (ANN)

For datasets larger than 1,000 items, Simile automatically builds an HNSW (Hierarchical Navigable Small World) index for near-instant search.

```typescript
const engine = await Simile.from(items, {
  useANN: true, // Force enable ANN
  annThreshold: 500, // Enable ANN if items > 500 (default: 1000)
});
```

### 🚀 Vector Caching

Avoid redundant AI embedding calls for duplicate texts with built-in LRU caching.

```typescript
const engine = await Simile.from(items, {
  cache: {
    maxSize: 5000, // Cache up to 5000 unique embeddings
    enableStats: true,
  }
});

// Check cache performance
const stats = engine.getIndexInfo().cacheStats;
console.log(`Cache Hit Rate: ${stats.hitRate}%`);
```

### 🔄 Non-blocking Background Updates

Adding items to a large index can be expensive. Simile uses an internal queue to process updates in the background without blocking search.

```typescript
// These return immediately/nearly immediately and process in batches
engine.add(newItems);
engine.add(moreItems);
```

---


<p align="center">
  Made with ❤️ by <a href="https://github.com/iaavas">Aavash Baral</a>
</p>

