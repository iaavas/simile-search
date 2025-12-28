/**
 * Extract a value from an object using a dot-notation path.
 * Supports nested paths like "author.firstName" and array access like "tags[0]".
 * 
 * @example
 * getByPath({ author: { firstName: "John" } }, "author.firstName") // "John"
 * getByPath({ tags: ["a", "b"] }, "tags[1]") // "b"
 * getByPath({ items: [{ name: "x" }] }, "items[0].name") // "x"
 */
export function getByPath(obj: any, path: string): any {
  if (!obj || !path) return undefined;

  // Handle array notation: convert "items[0].name" to "items.0.name"
  const normalizedPath = path.replace(/\[(\d+)\]/g, ".$1");
  const keys = normalizedPath.split(".");

  let current = obj;
  for (const key of keys) {
    if (current === null || current === undefined) {
      return undefined;
    }
    current = current[key];
  }

  return current;
}

/**
 * Extract searchable text from an item using configured paths.
 * If paths are provided, extracts and joins values from those paths.
 * Otherwise, returns the item's 'text' field directly.
 * 
 * @example
 * // With paths
 * extractText(
 *   { id: "1", text: "", metadata: { author: { name: "John" }, title: "Hello" } },
 *   ["metadata.author.name", "metadata.title"]
 * ) // "John Hello"
 * 
 * // Without paths
 * extractText({ id: "1", text: "Hello World" }) // "Hello World"
 */
export function extractText(item: any, paths?: string[]): string {
  if (!paths || paths.length === 0) {
    return item.text || "";
  }

  const parts: string[] = [];
  for (const path of paths) {
    const value = getByPath(item, path);
    if (value !== null && value !== undefined) {
      if (Array.isArray(value)) {
        parts.push(value.filter((v) => v != null).join(" "));
      } else {
        parts.push(String(value));
      }
    }
  }

  return parts.join(" ").trim();
}

/**
 * Normalize a score to a 0-1 range using min-max normalization.
 * Handles edge cases where min equals max.
 */
export function normalizeScore(
  value: number,
  min: number,
  max: number,
  floorMax: number = 0
): number {
  const effectiveMax = Math.max(max, floorMax);
  if (effectiveMax <= min) return value > 0 ? 1 : 0;
  return Math.max(0, Math.min(1, (value - min) / (effectiveMax - min)));
}
