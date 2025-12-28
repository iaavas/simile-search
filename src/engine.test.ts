import { describe, it, expect } from "vitest";
import { Simile } from "./engine";

describe("simile search", () => {
  it("returns semantically similar items", async () => {
    const engine = await Simile.from([
      {
        id: "1",
        text: "Bathroom floor cleaner",
        metadata: { category: "Cleaning" },
      },
      {
        id: "2",
        text: "Dishwashing liquid",
        metadata: { category: "Kitchen" },
      },
      {
        id: "3",
        text: "Ipod Charger",
        metadata: { categoryq: "Electronics" },
      },
    ]);

    const results = await engine.search("cleaner");
    console.log(results);

    expect(results.length).toBeGreaterThan(0);
    expect(results[0].id).toBe("1");
    expect(results[0].score).toBeGreaterThan(0.5);
  }, 30000);

  it("performance test: 10K items should search in <100ms", async () => {
    // Generate 10K test items
    const items = Array.from({ length: 10000 }, (_, i) => ({
      id: `item-${i}`,
      text: `Product ${i} - ${
        [
          "cleaner",
          "charger",
          "liquid",
          "cable",
          "headphones",
          "keyboard",
          "mouse",
          "monitor",
        ][i % 8]
      }`,
      metadata: { category: ["Electronics", "Cleaning", "Kitchen"][i % 3] },
    }));

    // Create engine with optimized ANN settings
    const engine = await Simile.from(items, {
      useANN: {
        efSearch: 20, // Fast search
        M: 16,
        efConstruction: 200,
      },
      annThreshold: 100, // Enable ANN early
    });

    // Verify ANN is enabled
    const info = engine.getIndexInfo();
    expect(info.type).toBe("hnsw");
    expect(info.size).toBe(10000);

    // Warm up: first search includes embedding time
    await engine.search("cleaner");

    // Performance test: search should be <100ms (excluding first-time embedding)
    const query = "phone charger";
    const startTime = performance.now();

    const results = await engine.search(query, {
      topK: 5,
      semanticOnly: true, // Fast mode: skip fuzzy/keyword
    });

    const endTime = performance.now();
    const searchTime = endTime - startTime;

    console.log(`Search time for 10K items: ${searchTime.toFixed(2)}ms`);
    console.log(`Results: ${results.length}`);
    console.log(`Index info:`, info);

    expect(results.length).toBeGreaterThan(0);
    expect(searchTime).toBeLessThan(100); // Should be <100ms

    // Also test with full hybrid search
    const startTime2 = performance.now();
    const results2 = await engine.search(query, {
      topK: 5,
      semanticOnly: false, // Full hybrid search
    });
    const endTime2 = performance.now();
    const hybridTime = endTime2 - startTime2;

    console.log(`Hybrid search time: ${hybridTime.toFixed(2)}ms`);
    expect(hybridTime).toBeLessThan(200);
  }, 300000); // Longer timeout for 10K items (embedding takes ~3 minutes)
});
