import "dotenv/config";
import express from "express";
import cors from "cors";
import memoryRoutes from "./routes/memory.js";

const app = express();
const DEFAULT_PORT = Number(process.env.PORT) || 3000;

app.use(cors());
app.use(express.json());

app.get("/health", (_, res) => res.json({ status: "ok" }));

app.use("/context", memoryRoutes);

app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(500).json({ error: "Internal server error", message: err?.message });
});

function tryListen(port) {
  const server = app.listen(port, () => {
    console.log(`Context API listening on http://localhost:${port}`);
  });
  server.on("error", (err) => {
    if (err.code === "EADDRINUSE") {
      console.warn(`Port ${port} in use, trying ${port + 1}...`);
      server.close(() => tryListen(port + 1));
    } else {
      throw err;
    }
  });
}

tryListen(DEFAULT_PORT);
