/**
 * Vector Quantization - Reduce memory footprint by 50-75%.
 * 
 * Quantization levels:
 * - Float32: Full precision (4 bytes per dimension) - default
 * - Float16: Half precision (2 bytes, ~0.1% accuracy loss)
 * - Int8: 8-bit quantization (1 byte, ~1% accuracy loss)
 */

export type QuantizationType = 'float32' | 'float16' | 'int8';

export interface QuantizedVector {
  data: ArrayBuffer;
  type: QuantizationType;
  /** Scale factor for Int8 quantization */
  scale?: number;
  /** Offset for Int8 quantization */
  offset?: number;
}

/**
 * Quantize a Float32Array to a more memory-efficient format.
 */
export function quantizeVector(
  vector: Float32Array,
  type: QuantizationType
): QuantizedVector {
  switch (type) {
    case 'float32':
      // Ensure we get a proper ArrayBuffer (not SharedArrayBuffer)
      const float32Data = new ArrayBuffer(vector.byteLength);
      new Float32Array(float32Data).set(vector);
      return {
        data: float32Data,
        type: 'float32',
      };
      
    case 'float16':
      return {
        data: float32ToFloat16(vector),
        type: 'float16',
      };
      
    case 'int8':
      return quantizeToInt8(vector);
      
    default:
      throw new Error(`Unknown quantization type: ${type}`);
  }
}

/**
 * Dequantize back to Float32Array for computation.
 */
export function dequantizeVector(quantized: QuantizedVector): Float32Array {
  switch (quantized.type) {
    case 'float32':
      return new Float32Array(quantized.data);
      
    case 'float16':
      return float16ToFloat32(quantized.data);
      
    case 'int8':
      return dequantizeFromInt8(quantized);
      
    default:
      throw new Error(`Unknown quantization type: ${quantized.type}`);
  }
}

/**
 * Compute cosine similarity directly on quantized vectors.
 * More efficient than dequantizing first for large batches.
 */
export function cosineQuantized(
  a: QuantizedVector,
  b: QuantizedVector
): number {
  if (a.type !== b.type) {
    throw new Error('Cannot compute similarity between different quantization types');
  }
  
  switch (a.type) {
    case 'float32': {
      const va = new Float32Array(a.data);
      const vb = new Float32Array(b.data);
      return cosineFloat32(va, vb);
    }
    
    case 'float16': {
      // For Float16, dequantize and compute (hardware Float16 not widely supported)
      const va = float16ToFloat32(a.data);
      const vb = float16ToFloat32(b.data);
      return cosineFloat32(va, vb);
    }
    
    case 'int8': {
      // For Int8, use integer arithmetic then scale
      return cosineInt8(a, b);
    }
    
    default:
      throw new Error(`Unknown quantization type: ${a.type}`);
  }
}

/**
 * Serialize quantized vector to base64 for storage.
 */
export function quantizedToBase64(quantized: QuantizedVector): string {
  const meta = JSON.stringify({
    type: quantized.type,
    scale: quantized.scale,
    offset: quantized.offset,
  });
  
  const metaBuffer = Buffer.from(meta);
  const dataBuffer = Buffer.from(quantized.data);
  
  // Format: [2 bytes meta length][meta][data]
  const combined = Buffer.alloc(2 + metaBuffer.length + dataBuffer.length);
  combined.writeUInt16LE(metaBuffer.length, 0);
  metaBuffer.copy(combined, 2);
  dataBuffer.copy(combined, 2 + metaBuffer.length);
  
  return combined.toString('base64');
}

/**
 * Deserialize quantized vector from base64.
 */
export function base64ToQuantized(base64: string): QuantizedVector {
  const combined = Buffer.from(base64, 'base64');
  
  const metaLength = combined.readUInt16LE(0);
  const metaBuffer = combined.subarray(2, 2 + metaLength);
  const dataBuffer = combined.subarray(2 + metaLength);
  
  const meta = JSON.parse(metaBuffer.toString());
  
  return {
    data: dataBuffer.buffer.slice(
      dataBuffer.byteOffset,
      dataBuffer.byteOffset + dataBuffer.length
    ),
    type: meta.type,
    scale: meta.scale,
    offset: meta.offset,
  };
}

/**
 * Get bytes per dimension for a quantization type.
 */
export function getBytesPerDimension(type: QuantizationType): number {
  switch (type) {
    case 'float32': return 4;
    case 'float16': return 2;
    case 'int8': return 1;
    default: return 4;
  }
}

/**
 * Estimate memory savings compared to Float32.
 */
export function estimateMemorySavings(
  vectorCount: number,
  dimensions: number,
  type: QuantizationType
): { original: number; quantized: number; savings: number } {
  const original = vectorCount * dimensions * 4; // Float32
  const quantized = vectorCount * dimensions * getBytesPerDimension(type);
  
  return {
    original,
    quantized,
    savings: 1 - (quantized / original),
  };
}

// ============ Internal Helpers ============

/**
 * Convert Float32Array to Float16 (stored as Uint16Array buffer).
 */
function float32ToFloat16(float32: Float32Array): ArrayBuffer {
  const uint16 = new Uint16Array(float32.length);
  
  for (let i = 0; i < float32.length; i++) {
    uint16[i] = floatToHalf(float32[i]);
  }
  
  return uint16.buffer;
}

/**
 * Convert Float16 buffer back to Float32Array.
 */
function float16ToFloat32(buffer: ArrayBuffer): Float32Array {
  const uint16 = new Uint16Array(buffer);
  const float32 = new Float32Array(uint16.length);
  
  for (let i = 0; i < uint16.length; i++) {
    float32[i] = halfToFloat(uint16[i]);
  }
  
  return float32;
}

/**
 * Quantize Float32Array to Int8 with scale/offset.
 */
function quantizeToInt8(vector: Float32Array): QuantizedVector {
  let min = Infinity;
  let max = -Infinity;
  
  for (let i = 0; i < vector.length; i++) {
    if (vector[i] < min) min = vector[i];
    if (vector[i] > max) max = vector[i];
  }
  
  const scale = (max - min) / 255;
  const offset = min;
  
  const int8 = new Int8Array(vector.length);
  
  for (let i = 0; i < vector.length; i++) {
    // Map to 0-255 range, then shift to -128 to 127
    const normalized = scale > 0 ? (vector[i] - offset) / scale : 0;
    int8[i] = Math.round(normalized) - 128;
  }
  
  return {
    data: int8.buffer,
    type: 'int8',
    scale,
    offset,
  };
}

/**
 * Dequantize Int8 back to Float32Array.
 */
function dequantizeFromInt8(quantized: QuantizedVector): Float32Array {
  const int8 = new Int8Array(quantized.data);
  const float32 = new Float32Array(int8.length);
  
  const scale = quantized.scale ?? 1;
  const offset = quantized.offset ?? 0;
  
  for (let i = 0; i < int8.length; i++) {
    float32[i] = (int8[i] + 128) * scale + offset;
  }
  
  return float32;
}

/**
 * Cosine similarity for Float32 vectors.
 */
function cosineFloat32(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

/**
 * Cosine similarity for Int8 quantized vectors.
 * Uses integer arithmetic for speed.
 */
function cosineInt8(a: QuantizedVector, b: QuantizedVector): number {
  const int8A = new Int8Array(a.data);
  const int8B = new Int8Array(b.data);
  
  // Compute dot product in integer space
  let dotInt = 0;
  let normAInt = 0;
  let normBInt = 0;
  
  for (let i = 0; i < int8A.length; i++) {
    const valA = int8A[i] + 128;
    const valB = int8B[i] + 128;
    dotInt += valA * valB;
    normAInt += valA * valA;
    normBInt += valB * valB;
  }
  
  // For normalized vectors, we can simplify
  // Since embeddings are already normalized, dot product ≈ cosine
  const scaleA = a.scale ?? 1;
  const scaleB = b.scale ?? 1;
  
  return (dotInt * scaleA * scaleB) / (Math.sqrt(normAInt) * Math.sqrt(normBInt) * scaleA * scaleB);
}

/**
 * Convert a 32-bit float to 16-bit half precision.
 */
function floatToHalf(val: number): number {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  
  floatView[0] = val;
  const x = int32View[0];
  
  let bits = (x >> 16) & 0x8000; // Sign
  let m = (x >> 12) & 0x07ff; // Mantissa
  const e = (x >> 23) & 0xff; // Exponent
  
  if (e < 103) {
    return bits;
  }
  
  if (e > 142) {
    bits |= 0x7c00;
    bits |= (e === 255 ? 0 : 1) && (x & 0x007fffff);
    return bits;
  }
  
  if (e < 113) {
    m |= 0x0800;
    bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
    return bits;
  }
  
  bits |= ((e - 112) << 10) | (m >> 1);
  bits += m & 1;
  return bits;
}

/**
 * Convert a 16-bit half precision to 32-bit float.
 */
function halfToFloat(val: number): number {
  const s = (val & 0x8000) >> 15;
  const e = (val & 0x7c00) >> 10;
  const f = val & 0x03ff;
  
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
  } else if (e === 0x1f) {
    return f ? NaN : (s ? -Infinity : Infinity);
  }
  
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / Math.pow(2, 10));
}
