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
- 💾 **Persistence** - Save/load embeddings to avoid re-computing
- ⚡ **Batch Processing** - Optimized for large catalogs
- 🔧 **Configurable** - Tune scoring weights for your use case
- 📦 **Zero API Calls** - Everything runs locally with Transformers.js

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
//   { id: '3', text: 'iPhone Charger', score: 0.72, ... },
//   { id: '4', text: 'USB-C phone charger cable', score: 0.68, ... },
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
//   createdAt: '2024-12-28T...'
// }

// Load from snapshot object
const restored = Simile.load(snapshot);
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

### Search Options

```typescript
const results = await engine.search('cleaner', {
  topK: 10,           // Max results to return (default: 5)
  threshold: 0.5,     // Minimum score (default: 0)
  explain: true,      // Include score breakdown
  filter: (meta) => meta.category === 'Cleaning',  // Filter by metadata
});

// With explain: true
// {
//   id: '1',
//   text: 'Bathroom floor cleaner',
//   score: 0.63,
//   explain: { semantic: 0.62, fuzzy: 0.32, keyword: 1.0 }
// }
```

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
  base64ToVector 
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
```

## 📊 API Reference

### `Simile.from(items, config?)`
Create a new engine from items. Embeds all items (async).

### `Simile.load(snapshot, config?)`
Load from a saved snapshot (instant, no embedding).

### `Simile.loadFromJSON(json, config?)`
Load from JSON string.

### `engine.search(query, options?)`
Search for similar items.

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
  explain?: { semantic: number; fuzzy: number; keyword: number };
}

interface SearchOptions {
  topK?: number;
  explain?: boolean;
  threshold?: number;
  filter?: (metadata: any) => boolean;
}

interface SimileConfig {
  weights?: { semantic?: number; fuzzy?: number; keyword?: number };
  model?: string;
}
```

## 🤖 Model

Simile uses [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) via Transformers.js by default. This model runs entirely in JavaScript—no Python or external APIs required.

## 📄 License

MIT © [Aavash Baral](https://github.com/iaavas)

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/iaavas">Aavash Baral</a>
</p>
