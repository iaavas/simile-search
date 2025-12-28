import { pipeline } from "@xenova/transformers";

let extractor: any;
let currentModel: string = "";

export async function getEmbedder(model: string = "Xenova/all-MiniLM-L6-v2") {
  if (!extractor || currentModel !== model) {
    extractor = await pipeline("feature-extraction", model);
    currentModel = model;
  }
  return extractor;
}

export async function embed(
  text: string,
  model?: string
): Promise<Float32Array> {
  const embedder = await getEmbedder(model);
  const output = await embedder(text, {
    pooling: "mean",
    normalize: true,
  });
  return output.data;
}

/**
 * Batch embed multiple texts at once for better performance.
 * This is significantly faster than embedding one by one.
 */
export async function embedBatch(
  texts: string[],
  model?: string
): Promise<Float32Array[]> {
  const embedder = await getEmbedder(model);
  const results: Float32Array[] = [];

  // Process in batches of 32 for memory efficiency
  const BATCH_SIZE = 32;
  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);
    const outputs = await Promise.all(
      batch.map((text) =>
        embedder(text, { pooling: "mean", normalize: true })
      )
    );
    results.push(...outputs.map((o: any) => o.data));
  }

  return results;
}

/** Serialize Float32Array to base64 string for storage */
export function vectorToBase64(vector: Float32Array): string {
  const buffer = Buffer.from(vector.buffer);
  return buffer.toString("base64");
}

/** Deserialize base64 string back to Float32Array */
export function base64ToVector(base64: string): Float32Array {
  const buffer = Buffer.from(base64, "base64");
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
}
