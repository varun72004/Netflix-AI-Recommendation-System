const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const PORT = Number(process.env.PORT || 3000);
const ROOT = __dirname;
const ARTIFACTS_DIR = path.join(ROOT, 'artifacts');
const MODEL_FILE = 'svd_model.joblib';
const LEGACY_MODEL_FILE = 'svd_model.pkl';
const DEFAULT_USER_ID = 1331154;

function loadDotEnv() {
  const envPath = path.join(ROOT, '.env');
  if (!fs.existsSync(envPath)) return;

  const lines = fs.readFileSync(envPath, 'utf8').split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) continue;
    const [key, ...valueParts] = trimmed.split('=');
    const value = valueParts.join('=').trim().replace(/^['"]|['"]$/g, '');
    if (key) {
      process.env[key.trim()] = value;
    }
  }
}

loadDotEnv();

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.joblib': 'application/octet-stream',
};

function sendJson(res, status, payload) {
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'access-control-allow-origin': '*',
    'access-control-allow-headers': 'content-type',
    'access-control-allow-methods': 'GET, POST, OPTIONS',
  });
  res.end(JSON.stringify(payload));
}

function readRequestBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => {
      body += chunk;
      if (body.length > 1_000_000) {
        req.destroy();
        reject(new Error('Request body is too large.'));
      }
    });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

async function handleRecommendations(req, res) {
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'access-control-allow-origin': '*',
      'access-control-allow-headers': 'content-type',
      'access-control-allow-methods': 'GET, POST, OPTIONS',
    });
    res.end();
    return;
  }

  if (req.method !== 'GET' && req.method !== 'POST') {
    sendJson(res, 405, { error: 'Use GET or POST for recommendations.' });
    return;
  }

  let payload;
  if (req.method === 'GET') {
    const url = new URL(req.url, `http://${req.headers.host}`);
    payload = {
      query: url.searchParams.get('query') || '',
      userId: url.searchParams.get('userId') || undefined,
      limit: url.searchParams.get('limit') || undefined,
    };
  } else {
    try {
      payload = JSON.parse(await readRequestBody(req));
    } catch {
      sendJson(res, 400, { error: 'Invalid JSON request body.' });
      return;
    }
  }

  const userId = getRequestedUserId(payload);
  const predictionPath = path.join(ARTIFACTS_DIR, `predictions_user_${userId}.json`);

  if (!fs.existsSync(predictionPath)) {
    sendJson(res, 404, {
      error: `No dumped SVD predictions found for user ${userId}.`,
      run: `cd "${ROOT}" && python dump_model.py --ratings combined_data_1.txt --user-id ${userId}`,
    });
    return;
  }

  try {
    const artifact = JSON.parse(fs.readFileSync(predictionPath, 'utf8'));
    const limit = Math.max(1, Math.min(Number(payload.limit || 5), 20));
    const modelInfo = getModelInfo();
    const hasModelDump = modelInfo.available;
    const titleQuery = getTitleQuery(payload);
    const recs = titleQuery
      ? buildMovieNameRecommendations(titleQuery, userId, limit, artifact, hasModelDump)
      : (artifact.recommendations || []).slice(0, limit);
    sendJson(res, 200, {
      recommendations: recs,
      ai_note: buildModelNote(artifact.user_id, hasModelDump, titleQuery),
      source: 'local-svd-artifact',
      user_id: artifact.user_id,
      model: modelInfo,
    });
  } catch (error) {
    sendJson(res, 500, { error: `Could not read model predictions: ${error.message}` });
  }
}

function handleModelStatus(req, res) {
  if (req.method !== 'GET') {
    sendJson(res, 405, { error: 'Use GET for model status.' });
    return;
  }

  sendJson(res, 200, getModelInfo());
}

function getModelInfo() {
  const preferredPath = path.join(ARTIFACTS_DIR, MODEL_FILE);
  const legacyPath = path.join(ARTIFACTS_DIR, LEGACY_MODEL_FILE);
  const modelPath = fs.existsSync(preferredPath) ? preferredPath : legacyPath;
  const available = fs.existsSync(modelPath);
  const manifestPath = path.join(ARTIFACTS_DIR, 'model_manifest.json');
  let manifest = null;

  if (fs.existsSync(manifestPath)) {
    try {
      manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    } catch {
      manifest = null;
    }
  }

  if (!available) {
    return {
      available: false,
      model_path: path.join('artifacts', MODEL_FILE),
      model_library: 'joblib',
      model_class: 'surprise.SVD',
      status: 'missing',
      run: `python dump_model.py --ratings combined_data_1.txt --user-id ${DEFAULT_USER_ID}`,
      manifest,
    };
  }

  const stats = fs.statSync(modelPath);
  return {
    available: true,
    model_path: path.relative(ROOT, modelPath).replace(/\\/g, '/'),
    model_library: path.basename(modelPath) === MODEL_FILE ? 'joblib' : 'surprise.dump',
    model_class: manifest?.model_class || 'surprise.SVD',
    status: 'ready',
    size_bytes: stats.size,
    size_label: formatBytes(stats.size),
    modified_at: stats.mtime.toISOString(),
    manifest,
  };
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return 'Unknown size';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getTitleQuery(payload) {
  const query = String(payload.query || payload.messages?.[0]?.content || '').trim();
  return /[A-Za-z]/.test(query) ? query : '';
}

function buildModelNote(userId, hasModelDump, titleQuery) {
  if (hasModelDump && titleQuery) {
    return `Movie-name search scored with the dumped Surprise SVD model for user ${userId}.`;
  }
  if (hasModelDump) {
    return `Local Surprise SVD model predictions for user ${userId}. Model artifact: artifacts/${MODEL_FILE}.`;
  }
  if (titleQuery) {
    return `Movie-name search is using title matches plus saved notebook predictions where available. Run dump_model.py with combined_data_1.txt to enable live SVD scoring for any title.`;
  }
  return `Saved SVD prediction output from Netflix_.ipynb for user ${userId}. Run dump_model.py with combined_data_1.txt to refresh the full model artifact.`;
}

function buildMovieNameRecommendations(titleQuery, userId, limit, artifact, hasModelDump) {
  const matches = searchMovieTitles(titleQuery, limit);
  if (matches.length === 0) {
    return (artifact.recommendations || []).slice(0, limit);
  }

  const artifactScores = new Map(
    (artifact.recommendations || []).map(item => [Number(item.movie_id), item])
  );
  const liveScores = hasModelDump
    ? predictScores(userId, matches.map(movie => movie.movie_id))
    : new Map();

  return matches.map((movie, index) => {
    const liveScore = liveScores.get(movie.movie_id);
    const saved = artifactScores.get(movie.movie_id);
    const estimate = liveScore ?? saved?.estimate_score ?? null;
    const score = estimate === null
      ? Math.max(55, 90 - index * 6)
      : Math.round((estimate / 5) * 100);
    return {
      movie_id: movie.movie_id,
      title: movie.title,
      year: movie.year,
      estimate_score: estimate,
      score,
      reason: estimate === null
        ? 'Title match; dump the model for live SVD score'
        : `SVD predicted ${estimate.toFixed(2)}/5 for user ${userId}`,
    };
  });
}

function searchMovieTitles(query, limit) {
  const csvPath = path.join(ROOT, 'movie_titles.csv');
  if (!fs.existsSync(csvPath)) return [];

  const normalizedQuery = normalizeTitle(query);
  const tokens = normalizedQuery.split(' ').filter(Boolean);
  const rows = fs.readFileSync(csvPath, 'latin1').split(/\r?\n/);
  const matches = [];

  for (const row of rows) {
    const movie = parseMovieTitleRow(row);
    if (!movie) continue;

    const normalizedTitle = normalizeTitle(movie.title);
    const containsAllTokens = tokens.every(token => normalizedTitle.includes(token));
    const exactBoost = normalizedTitle === normalizedQuery ? 100 : 0;
    const startsBoost = normalizedTitle.startsWith(normalizedQuery) ? 40 : 0;
    if (!containsAllTokens && exactBoost === 0 && startsBoost === 0) continue;

    matches.push({
      ...movie,
      matchScore: exactBoost + startsBoost + tokens.length * 10 - Math.abs(normalizedTitle.length - normalizedQuery.length) / 10,
    });
  }

  return matches
    .sort((a, b) => b.matchScore - a.matchScore)
    .slice(0, limit);
}

function parseMovieTitleRow(row) {
  if (!row.trim()) return null;
  const firstComma = row.indexOf(',');
  const secondComma = row.indexOf(',', firstComma + 1);
  if (firstComma === -1 || secondComma === -1) return null;

  const movieId = Number(row.slice(0, firstComma));
  const year = Number(row.slice(firstComma + 1, secondComma));
  const title = row.slice(secondComma + 1).trim();
  if (!Number.isFinite(movieId) || !title) return null;

  return {
    movie_id: movieId,
    year: Number.isFinite(year) ? year : null,
    title,
  };
}

function normalizeTitle(value) {
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function predictScores(userId, movieIds) {
  if (movieIds.length === 0) return new Map();

  const result = spawnSync(
    'python',
    [
      path.join(ROOT, 'predict_model.py'),
      '--user-id',
      String(userId),
      '--movie-ids',
      movieIds.join(','),
    ],
    { cwd: ROOT, encoding: 'utf8' }
  );

  if (result.status !== 0) return new Map();

  try {
    const parsed = JSON.parse(result.stdout);
    return new Map(
      (parsed.predictions || []).map(item => [Number(item.movie_id), Number(item.estimate_score)])
    );
  } catch {
    return new Map();
  }
}

function getRequestedUserId(payload) {
  if (Number.isInteger(Number(payload.userId))) {
    return Number(payload.userId);
  }

  const query =
    payload.query ||
    payload.messages?.[0]?.content ||
    '';
  const match = String(query).match(/\b(?:user\s*(?:id)?\s*[:#-]?\s*)?(\d{4,})\b/i);
  return match ? Number(match[1]) : DEFAULT_USER_ID;
}

function handleStatic(req, res) {
  const urlPath = decodeURIComponent(new URL(req.url, `http://${req.headers.host}`).pathname);
  const requested = urlPath === '/' ? '/netflix_ai_platform.html' : urlPath;
  const filePath = path.resolve(ROOT, `.${requested}`);

  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  fs.readFile(filePath, (error, data) => {
    if (error) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }

    const contentType = MIME_TYPES[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
    const headers = { 'content-type': contentType };
    if (path.extname(filePath).toLowerCase() === '.html') {
      headers['cache-control'] = 'no-store';
    }
    res.writeHead(200, headers);
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  if (req.url.startsWith('/api/recommendations')) {
    handleRecommendations(req, res);
    return;
  }

  if (req.url.startsWith('/api/model')) {
    handleModelStatus(req, res);
    return;
  }

  handleStatic(req, res);
});

server.listen(PORT, () => {
  console.log(`CineAI server running at http://localhost:${PORT}`);
});
