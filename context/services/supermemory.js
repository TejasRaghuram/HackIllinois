const SUPERMEMORY_API_URL = "https://api.supermemory.ai";
const CONTAINER_TAG = "incidents";

function buildIncidentContent(incident) {
  const parts = [];
  if (incident.name) parts.push(`Reported by ${incident.name}`);
  if (incident.address) parts.push(`at ${incident.address}`);
  if (incident.incidentType) parts.push(`Incident type: ${incident.incidentType}`);
  if (incident.severity) parts.push(`Severity: ${incident.severity}`);
  if (incident.notes) parts.push(`Notes: ${incident.notes}`);
  if (incident.status) parts.push(`Status: ${incident.status}`);
  if (incident.deployedResources?.length)
    parts.push(`Deployed resources: ${incident.deployedResources.join(", ")}`);
  return parts.length ? parts.join(". ") : "Incident record.";
}

function buildMemoryMetadata(incident, incidentId) {
  const meta = {
    incidentId,
    ...(incident.name != null && { name: String(incident.name) }),
    ...(incident.phoneNumber != null && { phoneNumber: String(incident.phoneNumber) }),
    ...(incident.address != null && { address: String(incident.address) }),
    ...(incident.incidentType != null && { incidentType: String(incident.incidentType) }),
    ...(incident.severity != null && { severity: String(incident.severity) }),
    ...(incident.notes != null && { notes: String(incident.notes) }),
    ...(incident.status != null && { status: String(incident.status) }),
  };
  if (incident.coordinates) {
    if (typeof incident.coordinates.lat === "number") meta.lat = incident.coordinates.lat;
    if (typeof incident.coordinates.lng === "number") meta.lng = incident.coordinates.lng;
  }
  if (Array.isArray(incident.deployedResources)) {
    meta.deployedResources = JSON.stringify(incident.deployedResources);
  }
  return meta;
}

function resultToIncident(result) {
  const meta = result.metadata || {};
  const incident = {
    id: meta.incidentId ?? result.id,
    name: meta.name,
    phoneNumber: meta.phoneNumber,
    address: meta.address,
    coordinates:
      meta.lat != null && meta.lng != null
        ? { lat: Number(meta.lat), lng: Number(meta.lng) }
        : undefined,
    incidentType: meta.incidentType,
    severity: meta.severity,
    notes: meta.notes,
    status: meta.status,
    deployedResources: undefined,
  };
  if (meta.deployedResources) {
    try {
      incident.deployedResources = JSON.parse(meta.deployedResources);
    } catch {
      incident.deployedResources = [];
    }
  }
  if (result.memory) incident.summary = result.memory;
  if (result.updatedAt) incident.updatedAt = result.updatedAt;
  return incident;
}

function getAuthHeader() {
  const raw = process.env.SUPERMEMORY_API_KEY;
  if (!raw) throw new Error("SUPERMEMORY_API_KEY is not set");
  const key = String(raw).trim();
  if (!key) throw new Error("SUPERMEMORY_API_KEY is empty");
  return { Authorization: `Bearer ${key}` };
}

let _client = null;
async function getClient() {
  if (!_client) {
    const raw = process.env.SUPERMEMORY_API_KEY;
    if (!raw) throw new Error("SUPERMEMORY_API_KEY is not set");
    const key = String(raw).trim();
    if (!key) throw new Error("SUPERMEMORY_API_KEY is empty");
    const { default: Supermemory } = await import("supermemory");
    _client = new Supermemory({ apiKey: key });
  }
  return _client;
}

export async function addContext(incident) {
  const { v4: uuidv4 } = await import("uuid");
  const incidentId = uuidv4();

  const content = buildIncidentContent(incident);
  const metadata = buildMemoryMetadata(incident, incidentId);

  const res = await fetch(`${SUPERMEMORY_API_URL}/v4/memories`, {
    method: "POST",
    headers: {
      ...getAuthHeader(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      containerTag: CONTAINER_TAG,
      memories: [{ content, metadata, isStatic: false }],
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    if (res.status === 401) {
      throw new Error(
        "Invalid or expired SUPERMEMORY_API_KEY. Check your .env: no extra spaces, and get a valid key at https://supermemory.ai"
      );
    }
    throw new Error(`Supermemory create failed: ${res.status} ${errText}`);
  }

  const data = await res.json();
  const memoryId = data.memories?.[0]?.id ?? null;
  return { id: incidentId, memoryId, documentId: data.documentId ?? null };
}

export async function searchContexts(query, options = {}) {
  const { limit = 20, threshold = 0.3, filters } = options;
  const client = await getClient();

  const params = {
    q: query || "*",
    containerTag: CONTAINER_TAG,
    searchMode: "memories",
    limit,
    threshold,
  };
  if (filters && Object.keys(filters).length) {
    params.filters = { AND: Object.entries(filters).map(([key, value]) => ({ key, value })) };
  }

  const response = await client.search.memories(params);
  const results = response.results ?? [];
  return results.map(resultToIncident);
}

export async function getContextById(incidentId) {
  const client = await getClient();
  const response = await client.search.memories({
    q: incidentId,
    containerTag: CONTAINER_TAG,
    searchMode: "memories",
    limit: 1,
    threshold: 0,
    filters: { AND: [{ key: "incidentId", value: incidentId }] },
  });
  const results = response.results ?? [];
  if (results.length === 0) return null;
  return resultToIncident(results[0]);
}

export async function updateContext(incidentId, updates) {
  const client = await getClient();
  const response = await client.search.memories({
    q: incidentId,
    containerTag: CONTAINER_TAG,
    searchMode: "memories",
    limit: 1,
    threshold: 0,
    filters: { AND: [{ key: "incidentId", value: incidentId }] },
  });
  const results = response.results ?? [];
  if (results.length === 0) {
    throw new Error(`Incident not found: ${incidentId}`);
  }
  const memoryId = results[0].id;
  const existing = resultToIncident(results[0]);

  const merged = {
    name: updates.name ?? existing.name,
    phoneNumber: updates.phoneNumber ?? existing.phoneNumber,
    address: updates.address ?? existing.address,
    coordinates: updates.coordinates ?? existing.coordinates,
    incidentType: updates.incidentType ?? existing.incidentType,
    severity: updates.severity ?? existing.severity,
    notes: updates.notes ?? existing.notes,
    status: updates.status ?? existing.status,
    deployedResources: updates.deployedResources ?? existing.deployedResources,
  };
  const content = buildIncidentContent(merged);
  const metadata = buildMemoryMetadata(merged, incidentId);

  const res = await fetch(`${SUPERMEMORY_API_URL}/v4/memories`, {
    method: "PATCH",
    headers: {
      ...getAuthHeader(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ id: memoryId, newContent: content, metadata }),
  });

  if (!res.ok) {
    const errText = await res.text();
    if (res.status === 401) {
      throw new Error(
        "Invalid or expired SUPERMEMORY_API_KEY. Check your .env: no extra spaces, and get a valid key at https://supermemory.ai"
      );
    }
    throw new Error(`Supermemory update failed: ${res.status} ${errText}`);
  }

  return getContextById(incidentId);
}
