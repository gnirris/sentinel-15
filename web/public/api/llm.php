<?php
/**
 * SENTINEL-15 LLM proxy. Lets visitors use the site's own LLM key without the
 * key ever reaching the browser. Speaks the OpenAI chat-completions format.
 *
 *   GET  llm.php/status            -> {"enabled": true, "model": "..."}
 *   GET  llm.php/models            -> {"data": [{"id": "..."}]}
 *   POST llm.php/chat/completions  -> streamed completion
 *
 * Config lives OUTSIDE the web root, in ~/sentinel-config.php (two levels up
 * from public_html/api/), so it can never be downloaded:
 *
 *   <?php return [
 *     'key'      => 'AQ....',                       // provider API key
 *     'model'    => 'gemini-3.5-flash-lite',
 *     'upstream' => 'https://generativelanguage.googleapis.com/v1beta/openai',
 *     'per_ip_hour' => 30,   // requests per visitor per hour
 *     'per_day'     => 600,  // requests per day for the whole site
 *   ];
 *
 * Abuse guard: this is not a general chatbot. The proxy always replaces the
 * system prompt with SENTINEL's own, requires the detector's <evidence> block,
 * caps sizes, and rate-limits per IP and per day.
 */
declare(strict_types=1);

const SYSTEM = <<<'TXT'
You are the explanation layer of SENTINEL-15, a network intrusion detector used by a security operations center (SOC). Two XGBoost models already made the decision; you never re-decide it. You translate their output for a human analyst.

Rules:
- Use ONLY the numbers in the evidence block. Never invent a field, value, IP address, host name, CVE or attacker. If the evidence does not say something, say it is unknown.
- When you cite a field, write its exact name in backticks (e.g. `sbytes`), then what it means.
- SHAP values are in log-odds: positive pushes toward INTRUSION, negative toward SAFE.
- The model sees flow metadata only (sizes, timings, counts, protocol), never packet contents. Do not claim to know payloads or intent.
- Be calibrated. If attack probability or category confidence is below 70%, say the call is uncertain. Mention that UNSW-NB15 is a 2015 lab dataset when it matters.
- Recommend investigation steps for a human. Never recommend automatic blocking or punishing a user based on this alert alone.
- Write for a smart 15-year-old: short sentences, no jargon without a gloss. Markdown, no headings above ###.
- Only discuss this detector, its evidence, network security and the model. Politely decline anything else.
TXT;

function fail(int $code, string $msg): never {
    http_response_code($code);
    header('Content-Type: application/json');
    echo json_encode(['error' => ['message' => $msg, 'code' => $code]]);
    exit;
}

$cfgFile = dirname(__DIR__, 2) . '/sentinel-config.php';
$cfg = is_file($cfgFile) ? (require $cfgFile) : [];
$enabled = !empty($cfg['key']) && !empty($cfg['model']) && !empty($cfg['upstream']);
$route = $_SERVER['PATH_INFO'] ?? '';
header('Cache-Control: no-store');

if ($route === '/status') {
    header('Content-Type: application/json');
    echo json_encode(['enabled' => $enabled, 'model' => $enabled ? $cfg['model'] : null]);
    exit;
}
if (!$enabled) fail(503, 'The site analyst is not configured.');
if ($route === '/models') {
    header('Content-Type: application/json');
    echo json_encode(['object' => 'list', 'data' => [['id' => $cfg['model'], 'object' => 'model']]]);
    exit;
}
if ($route !== '/chat/completions' || $_SERVER['REQUEST_METHOD'] !== 'POST') fail(404, 'Not found.');

// ── rate limits (files in the system temp dir) ──────────────────────────────
function bump(string $name, int $window, int $limit): bool {
    $f = sys_get_temp_dir() . '/sentinel15_' . hash('sha256', $name);
    $h = fopen($f, 'c+');
    if (!$h) return true;
    flock($h, LOCK_EX);
    $now = time();
    $hits = array_filter(array_map('intval', explode(',', stream_get_contents($h) ?: '')),
                         fn($t) => $t > $now - $window);
    $ok = count($hits) < $limit;
    if ($ok) $hits[] = $now;
    ftruncate($h, 0); rewind($h); fwrite($h, implode(',', $hits));
    flock($h, LOCK_UN); fclose($h);
    return $ok;
}
$ip = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
if (!bump("ip:$ip", 3600, (int)($cfg['per_ip_hour'] ?? 30)))
    fail(429, 'Hourly limit for the site analyst reached. Try again later, or paste your own free key in Settings.');
if (!bump('day', 86400, (int)($cfg['per_day'] ?? 600)))
    fail(429, "Today's limit for the site analyst is used up. Paste your own free key in Settings.");

// ── validate and rebuild the request ────────────────────────────────────────
$raw = file_get_contents('php://input', false, null, 0, 120000);
$in = json_decode($raw ?: '', true);
if (!is_array($in) || !isset($in['messages']) || !is_array($in['messages'])) fail(400, 'Bad request.');

$msgs = [['role' => 'system', 'content' => SYSTEM]];
foreach (array_slice($in['messages'], -12) as $m) {
    if (!is_array($m) || !in_array($m['role'] ?? '', ['user', 'assistant'], true)) continue;  // drops client system prompts
    $c = $m['content'] ?? '';
    if (!is_string($c)) fail(400, 'Bad message content.');
    $msgs[] = ['role' => $m['role'], 'content' => mb_substr($c, 0, 8000)];
}
$last = end($msgs);
if (($last['role'] ?? '') !== 'user' || !str_contains($last['content'], '<evidence>'))
    fail(400, 'The site analyst only answers questions about the detector.');

$body = json_encode(['model' => $cfg['model'], 'messages' => $msgs, 'stream' => true, 'max_tokens' => 1500]);

// ── stream from upstream to the browser ─────────────────────────────────────
@ini_set('zlib.output_compression', '0');
@ini_set('output_buffering', '0');
while (ob_get_level() > 0) ob_end_flush();
header('Content-Type: text/event-stream');
header('X-Accel-Buffering: no');

$status = 0;
$ch = curl_init(rtrim($cfg['upstream'], '/') . '/chat/completions');
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => $body,
    CURLOPT_HTTPHEADER => ['Content-Type: application/json', 'Authorization: Bearer ' . $cfg['key']],
    CURLOPT_TIMEOUT => 90,
    CURLOPT_HEADERFUNCTION => function ($ch, $line) use (&$status) {
        if (preg_match('#^HTTP/\S+\s+(\d{3})#', $line, $m)) {
            $status = (int)$m[1];
            http_response_code($status);
            if ($status >= 400) header('Content-Type: application/json');
        }
        return strlen($line);
    },
    CURLOPT_WRITEFUNCTION => function ($ch, $chunk) {
        echo $chunk;
        flush();
        return strlen($chunk);
    },
]);
if (!curl_exec($ch) && $status === 0) {
    header('Content-Type: application/json');
    fail(502, 'The site analyst could not reach the LLM provider.');
}
curl_close($ch);
