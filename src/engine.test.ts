import { describe, it, expect } from "vitest";
import { Simile } from "./engine";
import * as fs from "fs";
import * as path from "path";

const testItems = [
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
    metadata: { category: "Electronics" },
  },
  {
    id: "4",
    text: "Kitchen cleaning spray",
    metadata: { category: "Cleaning" },
  },
  {
    id: "5",
    text: "USB-C phone charger cable",
    metadata: { category: "Electronics" },
  },
];

describe("simile search", () => {
  it("returns semantically similar items", async () => {
    const engine = await Simile.from(testItems.slice(0, 3));

    const results = await engine.search("cleaner", { explain: true });
    console.log("Search for 'cleaner':", results);

    expect(results.length).toBeGreaterThan(0);
    expect(results[0].id).toBe("1");
    expect(results[0].score).toBeGreaterThan(0.5);
  }, 30000);

  it("differentiates between unrelated items", async () => {
    const engine = await Simile.from(testItems);

    // Search for "phone charger" - should clearly prefer electronics
    const results = await engine.search("phone charger", { explain: true });
    console.log("Search for 'phone charger':", results);

    // Both chargers should be in top 2 (order may vary based on model)
    const topTwoIds = [results[0].id, results[1].id];
    expect(topTwoIds).toContain("5"); // USB-C phone charger
    expect(topTwoIds).toContain("3"); // iPod Charger
    
    // Both chargers should score significantly higher than cleaning products
    const chargerScores = results.filter((r) =>
      r.metadata?.category === "Electronics"
    );
    const cleaningScores = results.filter((r) =>
      r.metadata?.category === "Cleaning"
    );

    // Electronics should score at least 0.4 higher than cleaning items
    expect(chargerScores[0].score).toBeGreaterThan(cleaningScores[0].score + 0.4);
  }, 30000);

  it("applies threshold filtering", async () => {
    const engine = await Simile.from(testItems);

    // With high threshold, should filter out low-scoring results
    const results = await engine.search("cleaner", { threshold: 0.5 });
    console.log("Search with threshold 0.5:", results);

    results.forEach((r) => {
      expect(r.score).toBeGreaterThanOrEqual(0.5);
    });
  }, 30000);
});

describe("simile persistence", () => {
  const snapshotPath = path.join(__dirname, "../.test-snapshot.json");

  it("saves and loads from snapshot", async () => {
    // Create engine and save
    const engine = await Simile.from(testItems);
    const snapshot = engine.save();

    expect(snapshot.version).toBe("0.2.0");
    expect(snapshot.items.length).toBe(5);
    expect(snapshot.vectors.length).toBe(5);
    expect(snapshot.model).toBe("Xenova/all-MiniLM-L6-v2");

    // Load from snapshot (instant - no embedding!)
    const loadedEngine = Simile.load(snapshot);
    expect(loadedEngine.size).toBe(5);

    // Search should work the same
    const results = await loadedEngine.search("cleaner");
    expect(results[0].text).toContain("cleaner");
  }, 30000);

  it("saves and loads from JSON file", async () => {
    // Create and save to file
    const engine = await Simile.from(testItems);
    const json = engine.toJSON();
    fs.writeFileSync(snapshotPath, json);

    // Load from file (instant!)
    const loadedJson = fs.readFileSync(snapshotPath, "utf-8");
    const loadedEngine = Simile.loadFromJSON(loadedJson);

    expect(loadedEngine.size).toBe(5);

    // Cleanup
    fs.unlinkSync(snapshotPath);
  }, 30000);
});

describe("simile dynamic items", () => {
  it("adds new items", async () => {
    const engine = await Simile.from(testItems.slice(0, 2));
    expect(engine.size).toBe(2);

    await engine.add([testItems[2], testItems[3]]);
    expect(engine.size).toBe(4);

    const results = await engine.search("charger");
    expect(results.some((r) => r.id === "3")).toBe(true);
  }, 30000);

  it("removes items", async () => {
    const engine = await Simile.from(testItems);
    expect(engine.size).toBe(5);

    engine.remove(["1", "2"]);
    expect(engine.size).toBe(3);
    expect(engine.get("1")).toBeUndefined();
    expect(engine.get("3")).toBeDefined();
  }, 30000);

  it("updates existing items", async () => {
    const engine = await Simile.from(testItems.slice(0, 2));

    // Update item with same ID but different text
    await engine.add([
      { id: "1", text: "Wireless Bluetooth headphones", metadata: { category: "Electronics" } },
    ]);

    expect(engine.size).toBe(2); // Still 2 items, not 3
    expect(engine.get("1")?.text).toBe("Wireless Bluetooth headphones");
  }, 30000);
});

describe("simile custom weights", () => {
  it("respects custom weights", async () => {
    // Engine with high semantic weight
    const semanticEngine = await Simile.from(testItems, {
      weights: { semantic: 0.9, fuzzy: 0.05, keyword: 0.05 },
    });

    // Engine with high keyword weight
    const keywordEngine = await Simile.from(testItems, {
      weights: { semantic: 0.1, fuzzy: 0.1, keyword: 0.8 },
    });

    const query = "floor";
    const semanticResults = await semanticEngine.search(query, { explain: true });
    const keywordResults = await keywordEngine.search(query, { explain: true });

    console.log("Semantic-weighted results:", semanticResults.map((r) => ({
      text: r.text,
      score: r.score,
    })));
    console.log("Keyword-weighted results:", keywordResults.map((r) => ({
      text: r.text,
      score: r.score,
    })));

    // Both should find floor cleaner first (it has "floor" in text)
    expect(semanticResults[0].text).toContain("floor");
    expect(keywordResults[0].text).toContain("floor");
  }, 30000);
});
