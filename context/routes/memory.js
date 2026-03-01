import { Router } from "express";
import * as supermemory from "../services/supermemory.js";

const router = Router();

function requireSupermemoryKey(req, res, next) {
  if (!process.env.SUPERMEMORY_API_KEY) {
    return res.status(503).json({
      error: "Service unavailable",
      message:
        "SUPERMEMORY_API_KEY is not configured. Create a .env file in the context folder with: SUPERMEMORY_API_KEY=your_api_key (get a key at https://supermemory.ai)",
    });
  }
  next();
}

router.use(requireSupermemoryKey);

router.post("/", async (req, res) => {
  try {
    const incident = req.body ?? {};
    const result = await supermemory.addContext(incident);
    return res.status(201).json(result);
  } catch (err) {
    const status = err.message?.includes("not set") ? 503 : 400;
    return res.status(status).json({
      error: status === 503 ? "Service unavailable" : "Bad request",
      message: err.message,
    });
  }
});

const SEARCH_PARAMS = new Set(["q", "limit", "threshold"]);

router.get("/search", async (req, res) => {
  try {
    const q = req.query.q ?? "*";
    const limit = Math.min(parseInt(req.query.limit, 10) || 20, 100);
    const threshold = parseFloat(req.query.threshold) || 0.3;
    const filters = {};
    for (const [key, value] of Object.entries(req.query)) {
      if (!SEARCH_PARAMS.has(key) && value != null && value !== "") {
        filters[key] = String(value);
      }
    }
    const results = await supermemory.searchContexts(q, { limit, threshold, filters });
    return res.json({ results });
  } catch (err) {
    const status = err.message?.includes("not set") ? 503 : 500;
    return res.status(status).json({
      error: status === 503 ? "Service unavailable" : "Internal server error",
      message: err.message,
    });
  }
});

router.get("/:id", async (req, res) => {
  try {
    const incident = await supermemory.getContextById(req.params.id);
    if (!incident) {
      return res.status(404).json({ error: "Not found", message: "Incident not found" });
    }
    return res.json(incident);
  } catch (err) {
    const status = err.message?.includes("not set") ? 503 : 500;
    return res.status(status).json({
      error: status === 503 ? "Service unavailable" : "Internal server error",
      message: err.message,
    });
  }
});

router.patch("/:id", async (req, res) => {
  try {
    const updated = await supermemory.updateContext(req.params.id, req.body ?? {});
    return res.json(updated);
  } catch (err) {
    if (err.message?.includes("not found")) {
      return res.status(404).json({ error: "Not found", message: err.message });
    }
    const status = err.message?.includes("not set") ? 503 : 400;
    return res.status(status).json({
      error: status === 503 ? "Service unavailable" : "Bad request",
      message: err.message,
    });
  }
});

export default router;
