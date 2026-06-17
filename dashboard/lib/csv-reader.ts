import fs from "fs";
import path from "path";
import Papa from "papaparse";

const DATA_DIR = path.join(process.cwd(), "..", "data");

const cache = new Map<string, { data: unknown[]; timestamp: number }>();
const CACHE_TTL = 30_000;

export function readCsv<T>(relativePath: string): T[] {
  const fullPath = path.join(DATA_DIR, relativePath);
  const now = Date.now();
  const cached = cache.get(fullPath);
  if (cached && now - cached.timestamp < CACHE_TTL) {
    return cached.data as T[];
  }
  if (!fs.existsSync(fullPath)) return [];
  const content = fs.readFileSync(fullPath, "utf-8");
  const result = Papa.parse<T>(content, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  cache.set(fullPath, { data: result.data, timestamp: now });
  return result.data;
}

export function readJson<T>(relativePath: string): T | null {
  const fullPath = path.join(DATA_DIR, relativePath);
  if (!fs.existsSync(fullPath)) return null;
  const content = fs.readFileSync(fullPath, "utf-8");
  return JSON.parse(content);
}
