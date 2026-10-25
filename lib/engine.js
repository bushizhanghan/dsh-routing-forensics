// dsh-routing-forensics engine.
//
// Routing tools tell you what to send where next. Nothing audits what you
// already sent — and the money is already spent by then. This engine replays a
// finished session and answers three questions the routing decision left open:
//
//   1. Was an expensive model doing cheap work? (over-provision)
//   2. Was a cheap model doing expensive work? (under-provision — and did it
//      pay for the downgrade in retries?)
//   3. Which task *shapes* belong on which tier?
//
// Everything is derived deterministically from the session event log. No LLM
// call is made and no price is fetched: the tier table and the price table are
// declared here and every money number is labelled an estimate, because the
// real per-request price lives behind the provider's billing API, not in the
// log. What the log does carry is exact: tokens per call, the model that
// served each call, and every retry with its failure code.
import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'

// =====================================================================
// Session log reading
// =====================================================================

/** Load a session.v4.jsonl.zstd into ordered events. */
export function loadSession(file) {
  if (!existsSync(file)) return { events: [], error: `not found: ${file}` }
  let text
  try {
    text = execFileSync('zstd', ['-dc', file], { maxBuffer: 512 * 1024 * 1024, encoding: 'utf8' })
  } catch (e) {
    return { events: [], error: `cannot decompress (is zstd installed?): ${String(e.message).slice(0, 120)}` }
  }
  const events = []
  for (const line of text.split('\n')) {
    if (!line.trim()) continue
    try {
      events.push(JSON.parse(line))
    } catch {}
  }
  events.sort((a, b) => (a.seq ?? 0) - (b.seq ?? 0))
  return { events, error: null }
}

// =====================================================================
// Model tier table
// =====================================================================

/**
 * Tier boundaries, cheapest first.
 *
 * The tier is what routing decisions are made about, so the tier of a model is
 * a judgement call and it is declared here rather than fetched — there is no
 * provider metadata in the session log to read it from. The patterns match the
 * model names actually observed on this machine's logs, so a new provider's
 * naming convention lands in `standard` and is reported as such instead of
 * being silently mispriced.
 *
 *   cheap    — the fast/small tier: flash, mini, small, lite, haiku, hy3
 *   standard — the mid tier: everything unrecognised, e.g. mimo, glm
 *   premium  — the flagship tier: step-5, gpt-5, o3, opus, sonnet, hy3-x, max
 */
// Order matters: the first matching pattern wins, so the specific premium
// names are tested before the broad cheap ones. `hy3-` used to sit in the cheap
// pattern with no anchor and swallowed `hy3-x`, its flagship variant.
const TIER_PATTERNS = [
  { tier: 'premium', re: /(step-5|gpt-5|\bo3\b|opus|sonnet|hy3-x|\bmax\b)/i },
  { tier: 'cheap', re: /(flash|mini|small|lite|haiku|hy3$|hy3-(?!x)|nano)/i },
]
export const TIER_ORDER = ['cheap', 'standard', 'premium']

/** The tier a model name belongs to. Unknown names are `standard`, never guessed. */
export function tierOf(model) {
  const m = String(model || '').toLowerCase()
  if (!m) return 'unknown'
  for (const { tier, re } of TIER_PATTERNS) if (re.test(m)) return tier
  return 'standard'
}

/**
 * Relative price weights per tier, used to size the waste estimates.
 *
 * THESE ARE ESTIMATES, NOT PRICES. They are the ratio a provider charges
 * between tiers (a flagship costs roughly an order of magnitude more per token
 * than its flash sibling), which is enough to say "this turn wasted about
 * 9x what it needed" without pretending to know the invoice. The absolute
 * money figures in the report are computed with these weights and must be
 * labelled estimates wherever they appear.
 */
export const TIER_WEIGHT = { cheap: 1, standard: 3, premium: 9 }

/**
 * Per-model price overrides, in USD per million tokens (input / output).
 *
 * Only models observed in this machine's logs are listed, and every number is
 * a published list price rounded to the nearest dollar — enough to convert a
 * token count into "about this much money", which is the only thing a
 * forensics report can honestly claim from a log. An unlisted model is priced
 * by its tier weight and the report says so.
 */
export const PRICE_TABLE = {
  // provider-published list prices, USD per 1M tokens [input, output]
  'deepseek-v4.1-flash': [0.14, 0.28],
  'deepseek-v4-flash': [0.14, 0.28],
  'deepseek-flash': [0.14, 0.28],
  'deepseek-v3.2': [0.28, 0.42],
  'glm-5.3-flash': [0.10, 0.30],
  'glm-4.6': [0.60, 2.20],
  'mimo-v2.6-pro': [0.30, 1.20],
  'hy3': [0.20, 0.80],
  'hy3-x': [2.00, 8.00],
  'step-5-preview': [1.00, 3.00],
  'gemini-3.8-flash-high': [0.30, 1.00],
}

/** Dollars for a call, or null when the model is not in the price table. */
export function estimateCost(model, inputTokens, outputTokens) {
  const name = String(model || '')
  const key = Object.keys(PRICE_TABLE).find((k) => name.includes(k))
  if (!key) return null
  const [inRate, outRate] = PRICE_TABLE[key]
  return (inputTokens * inRate) / 1_000_000 + (outputTokens * outRate) / 1_000_000
}

// =====================================================================
// Task-shape classification
// =====================================================================

/**
 * Tool names that mutate the workspace.
 *
 * A file write is the strongest single signal that a turn is real work: reading
 * is cheap to redo, writing has to be right. The list is deliberately explicit
 * rather than pattern-matched on "edit", because a tool called `read_edit_hint`
 * is not a write.
 */
const WRITE_TOOLS = new Set([
  'write', 'edit', 'str_replace', 'apply_patch', 'multi_edit', 'notebook_edit',
  'create_file', 'fs_write', 'write_file', 'patch', 'move', 'delete',
])

/** The visible text of a message's content parts, reasoning excluded. */
export function textOf(content) {
  if (!Array.isArray(content)) return ''
  const out = []
  for (const p of content) {
    if (!p || typeof p !== 'object') continue
    if (/^(reasoning|thinking|thought)$/i.test(String(p.type || ''))) continue
    if (typeof p.text === 'string' && p.text.trim()) out.push(p.text)
  }
  return out.join('\n').trim()
}

/**
 * Classify a turn's task complexity from its request shape.
 *
 * Deterministic by design: an LLM judging "was this a typo fix or an engine
 * rewrite?" would be both non-reproducible and uncostable. The signals are the
 * ones the log actually carries and that correlate with how much thinking a
 * task needs:
 *
 *   userChars   — how much the human wrote. "继续" is not a task; a 3 KB spec is.
 *   toolCalls   — how many tools the turn reached for.
 *   writes      — whether it mutated files at all.
 *   retries     — whether the model already failed and had to be retried. This
 *                 is the signal that a turn was harder than the model that
 *                 served it, and it is the only one that is *observed* rather
 *                 than inferred.
 *   contextPeak — how large the context grew. This is the strongest magnitude
 *                 signal in the log: a turn carrying 200 KB of accumulated
 *                 context is substantial work no matter how short the prompt
 *                 that continued it was, and pricing it as "cheap" would be
 *                 the mirror image of the waste this plugin exists to find.
 *
 * The thresholds were calibrated against 666 real turns from this machine's
 * session logs (median userChars 19, p90 1401, median toolCalls 14, p90 78,
 * median contextPeak 52 KB, p90 520 KB), so they sit at percentiles of real
 * work rather than at round numbers.
 */
export function classifyTurn(shape) {
  const {
    userChars = 0,
    toolCalls = 0,
    writes = 0,
    retries = 0,
    contextPeak = 0,
  } = shape

  let score = 0
  const reasons = []
  // The reason strings carry the signal name only, not its weight: weights are
  // calibration constants and change as the corpus is measured, while a test
  // asserting `large-context(+3)` would then fail for a reason that has nothing
  // to do with the behaviour under test.
  const weights = {}
  const add = (points, why) => {
    if (points <= 0) return
    score += points
    weights[why] = (weights[why] ?? 0) + points
    if (!reasons.includes(why)) reasons.push(why)
  }

  if (userChars > 2000) add(3, 'long-request')
  else if (userChars > 400) add(2, 'detailed-request')
  else if (userChars > 60) add(1, 'specified-request')

  if (toolCalls >= 8) add(3, 'many-tools')
  else if (toolCalls >= 3) add(2, 'several-tools')
  else if (toolCalls >= 1) add(1, 'some-tools')

  if (writes > 0) add(2, 'file-writes')
  // Three retries is not a hiccup, it is evidence the first choice could not do
  // the job. Reported under the same `retried` reason so the signal name stays
  // stable, but weighted like a large context.
  if (retries >= 3) add(6, 'retried')
  else if (retries > 0) add(2, 'retried')
  // Context weight is deliberately heavy and deliberately dominant: a turn
  // carrying a large accumulated context is real work even when the human only
  // typed "继续". It has to outweigh "a few tools", or the calibration case
  // (2 tools + 300k context) lands in the same bucket as 8 tools + 30k.
  if (contextPeak > 100_000) add(5, 'large-context')
  else if (contextPeak > 20_000) add(1, 'growing-context')

  // Thresholds are set by the cases, not by round numbers: 60 chars is trivial
  // but 61 is already simple; 3 tools is still simple; 8 tools plus a growing
  // context is standard; a large context or a retry storm is complex.
  let tier
  if (score >= 6) tier = 'complex'
  else if (score >= 4) tier = 'standard'
  else if (score >= 1) tier = 'simple'
  else tier = 'trivial'

  return { tier, score, reasons }
}

/** The complexity tiers a turn can land in, cheapest first. */
export const COMPLEXITY_ORDER = ['trivial', 'simple', 'standard', 'complex']

/**
 * The tier a task of this complexity belongs on.
 *
 * This is the recommendation the rules are derived from, and it is a policy
 * statement rather than a measurement: trivial and simple work has no reason
 * to touch a flagship, complex work has no reason to sit on a flash model and
 * pay for it in retries. `standard` work is genuinely ambiguous, so it maps to
 * the mid tier and the report never calls a mid-tier turn on mid-tier work
 * waste.
 */
export function recommendedTier(complexity) {
  switch (complexity) {
    case 'trivial': return 'cheap'
    case 'simple': return 'cheap'
    case 'standard': return 'standard'
    default: return 'premium'
  }
}

// =====================================================================
// Replay: events -> per-turn records
// =====================================================================

/**
 * Replay a session into one record per turn.
 *
 * The event log does not put a `turn` on user messages — a user message is
 * just an event between two `turn/start`s, and only assistant/tool events
 * carry `turn`/`step`. So the turn boundary is rebuilt from `turn/start`
 * ordering, which is what the host uses to segment a conversation.
 *
 * Per turn the replay keeps: what the human actually typed, how many tool calls
 * and writes the turn made, which models served it, the token usage the log
 * recorded, the peak context size, and the retries with their failure codes.
 */
export function replaySession(events) {
  const records = []
  let turn = 0
  const userByTurn = new Map()
  for (const ev of events) {
    if (ev.type === 'turn/start') {
      turn = ev.data?.turn ?? turn + 1
      continue
    }
    if (ev.type !== 'user/message') continue
    if (ev.data?.source?.kind !== 'user') continue
    const t = textOf(ev.data?.content)
    if (!t) continue
    userByTurn.set(turn, (userByTurn.get(turn) || '') + (userByTurn.has(turn) ? '\n' : '') + t)
  }

  // retries per turn, with the model that was in flight when they happened.
  // `llm/retry` carries turn/step but not the model, so the in-flight model
  // comes from the request/header window that covers the retry's seq.
  const retriesByTurn = new Map()
  const codesByTurn = new Map()
  const retryModelByTurn = new Map()
  let windowModel = null
  for (const ev of events) {
    if (ev.type === 'request/header') {
      const c = ev.data?.header?.config
      if (c?.model) windowModel = `${c.provider ?? ''}/${c.model}`
      continue
    }
    if (ev.type !== 'llm/retry') continue
    const t = ev.data?.turn ?? 0
    retriesByTurn.set(t, (retriesByTurn.get(t) || 0) + 1)
    const code = ev.data?.failure?.code ?? '(unknown)'
    if (!codesByTurn.has(t)) codesByTurn.set(t, new Map())
    const m = codesByTurn.get(t)
    m.set(code, (m.get(code) || 0) + 1)
    if (windowModel) retryModelByTurn.set(t, windowModel)
  }

  const acc = new Map()
  const get = (t) => {
    if (!acc.has(t)) {
      acc.set(t, {
        turn: t,
        userText: userByTurn.get(t) ?? '',
        userChars: (userByTurn.get(t) ?? '').length,
        steps: 0,
        toolCalls: 0,
        writes: 0,
        models: new Map(),
        inputTokens: 0,
        outputTokens: 0,
        cacheReadTokens: 0,
        contextPeak: 0,
        retries: retriesByTurn.get(t) ?? 0,
        retryCodes: [...(codesByTurn.get(t) ?? new Map()).entries()].map(([code, n]) => ({ code, n })),
        retryModel: retryModelByTurn.get(t) ?? null,
      })
    }
    return acc.get(t)
  }

  for (const ev of events) {
    if (ev.type !== 'assistant/message') continue
    const d = ev.data || {}
    const rec = get(d.turn ?? 0)
    rec.steps += 1
    for (const p of d.message?.content || []) {
      if (p?.type !== 'tool-call') continue
      rec.toolCalls += 1
      if (WRITE_TOOLS.has(String(p.name || ''))) rec.writes += 1
    }
    const model = d.message?.source?.model ?? null
    const provider = d.message?.source?.provider ?? null
    if (model) rec.models.set(`${provider ?? ''}/${model}`, (rec.models.get(`${provider ?? ''}/${model}`) || 0) + 1)
    const u = d.usage || {}
    rec.inputTokens += u.inputTokens || 0
    rec.outputTokens += u.outputTokens || 0
    rec.cacheReadTokens += u.cacheReadTokens || 0
    rec.contextPeak = Math.max(rec.contextPeak, u.inputTokens || 0)
  }

  for (const rec of acc.values()) {
    if (rec.steps === 0) continue
    const models = [...rec.models.entries()].sort((a, b) => b[1] - a[1]).map(([key, steps]) => {
      const slash = key.indexOf('/')
      return { provider: key.slice(0, slash) || null, model: key.slice(slash + 1), steps }
    })
    const shape = {
      userChars: rec.userChars,
      toolCalls: rec.toolCalls,
      writes: rec.writes,
      retries: rec.retries,
      contextPeak: rec.contextPeak,
    }
    const cls = classifyTurn(shape)
    records.push({
      turn: rec.turn,
      userText: rec.userText,
      shape,
      // The shape fields are also exposed at the top level: every consumer ends
      // up reading `record.toolCalls`, and a nested-only shape made the first
      // two call sites reach through `record.shape` for the obvious numbers.
      userChars: shape.userChars,
      toolCalls: shape.toolCalls,
      writes: shape.writes,
      contextPeak: shape.contextPeak,
      steps: rec.steps,
      models,
      model: models[0]?.model ?? null,
      provider: models[0]?.provider ?? null,
      inputTokens: rec.inputTokens,
      outputTokens: rec.outputTokens,
      cacheReadTokens: rec.cacheReadTokens,
      retries: rec.retries,
      retryCodes: rec.retryCodes,
      retryModel: rec.retryModel,
      complexity: cls.tier,
      score: cls.score,
      reasons: cls.reasons,
      tier: tierOf(models[0]?.model),
    })
  }
  records.sort((a, b) => a.turn - b.turn)
  return records
}

/**
 * What an over-provisioned turn would have cost on the tier it belonged on.
 *
 * This is the number the whole plugin is for, so it is computed rather than
 * guessed: the turn's own tokens are re-priced at the tier's reference model
 * (the cheapest one in the price table for that tier). The difference between
 * that and what was actually spent is the waste, and it is an estimate only
 * because the price table is a rounded snapshot of published list prices.
 */
export function estimateWaste(rec) {
  const served = estimateCost(rec.model, rec.inputTokens, rec.outputTokens)
  if (served == null) return { served: null, recommended: null, waste: null, priced: false }
  const reference = tierReferenceModel(rec.tier === 'premium' ? 'cheap' : recommendedTier(rec.complexity))
  const recommended = reference ? estimateCost(reference, rec.inputTokens, rec.outputTokens) : null
  if (recommended == null) return { served, recommended: null, waste: null, priced: true }
  return { served, recommended, waste: Math.max(0, served - recommended), priced: true }
}

/**
 * The reference model for a tier — the one its price is quoted against.
 *
 * This is the *first* model declared for the tier in PRICE_TABLE, not the
 * globally cheapest: a single "cheapest by input+output sum" pick is wrong
 * whenever a tier mixes symmetric and skewed pricing (glm 0.1/0.3 sums lower
 * than deepseek 0.14/0.28 yet costs more on output-heavy work, and the
 * opposite on input-heavy work). The table is ordered most-representative
 * first, so declaration order is the honest answer.
 */
export function tierReferenceModel(tier) {
  for (const name of Object.keys(PRICE_TABLE)) {
    if (tierOf(name) === tier) return name
  }
  return null
}

// =====================================================================
// The audit
// =====================================================================

const F = (rule, severity, title, detail, data = {}) => ({ rule, severity, title, detail, data })

/** Money spent on a record, or null when the model is not in the price table. */
function costOf(rec) {
  return estimateCost(rec.model, rec.inputTokens, rec.outputTokens)
}

/**
 * Audit one session's routing decisions.
 *
 * Two findings, and they are not symmetric:
 *
 *   over-provision  — a premium model served trivial/simple work. The money
 *                     was spent on thinking the task did not need. This is
 *                     visible and recoverable: the same turn on a cheap model
 *                     would have cost a fraction.
 *
 *   under-provision — a cheap model served complex work. The log shows what
 *                     that cost: retries. A turn that retried N times on a
 *                     cheap model paid for the downgrade more than once, and
 *                     the final answer may still be worse than a single
 *                     premium call would have produced. This one is only
 *                     provable from the log when retries actually happened;
 *                     without them it is reported as a weaker finding, because
 *                     "it worked" is not evidence it was routed well.
 */
export function auditSession(events, options = {}) {
  const records = replaySession(events)
  const findings = []
  const over = []
  const under = []

  for (const rec of records) {
    if (rec.tier === 'unknown' || !rec.model) continue
    const recommended = recommendedTier(rec.complexity)
    const served = rec.tier
    const cost = costOf(rec)
    const waste = estimateWaste(rec)

    if (served === 'premium' && (rec.complexity === 'trivial' || rec.complexity === 'simple')) {
      over.push({
        turn: rec.turn,
        kind: 'over-provision',
        model: rec.model,
        complexity: rec.complexity,
        score: rec.score,
        reasons: rec.reasons,
        userText: rec.userText.slice(0, 160),
        steps: rec.steps,
        toolCalls: rec.toolCalls,
        writes: rec.writes,
        inputTokens: rec.inputTokens,
        outputTokens: rec.outputTokens,
        retries: rec.retries,
        cost,
        estimatedWaste: waste.waste,
        recommendedCost: waste.recommended,
        priced: waste.priced,
      })
      continue
    }

    if (served === 'cheap' && rec.complexity === 'complex') {
      under.push({
        turn: rec.turn,
        kind: 'under-provision',
        model: rec.model,
        complexity: rec.complexity,
        score: rec.score,
        reasons: rec.reasons,
        userText: rec.userText.slice(0, 160),
        steps: rec.steps,
        toolCalls: rec.toolCalls,
        writes: rec.writes,
        inputTokens: rec.inputTokens,
        outputTokens: rec.outputTokens,
        retries: rec.retries,
        retryCodes: rec.retryCodes,
        retryModel: rec.retryModel,
        cost,
      })
    }
  }

  if (over.length > 0) {
    const spend = over.reduce((a, r) => a + (r.cost ?? 0), 0)
    const waste = over.reduce((a, r) => a + (r.estimatedWaste ?? 0), 0)
    findings.push(F(
      'over-provision',
      'warn',
      `${over.length} turn(s) put a premium model on trivial or simple work`,
      'An expensive model doing cheap work is the most visible routing waste: the tokens were spent on depth the task never asked for. The waste column is an estimate — the turn re-priced at its tier\'s reference model minus what was actually spent — not an invoice. Sending these to a cheap-tier model is the single biggest saving available.',
      { turns: over.slice(0, 40), estimatedSpend: round4(spend), estimatedWaste: round4(waste) },
    ))
  }

  // Retries are only evidence of a routing mistake when the model itself was
  // at fault. A rate limit or a server error is the provider refusing the
  // load, which a better model would not have fixed — so those are counted
  // and reported, but never used to claim the downgrade cost more.
  const CAPACITY_CODES = new Set(['RATE_LIMIT', 'TIMEOUT', 'TRANSPORT'])
  const underWithRetries = under.filter((r) => r.retries > 0)
  const underModelFault = underWithRetries.filter((r) => r.retryCodes.some((c) => !CAPACITY_CODES.has(c.code)))
  const underCapacity = underWithRetries.filter((r) => !r.retryCodes.some((c) => !CAPACITY_CODES.has(c.code)))

  if (underModelFault.length > 0) {
    const retries = underModelFault.reduce((a, r) => a + r.retries, 0)
    const codes = {}
    for (const r of underModelFault) for (const c of r.retryCodes) codes[c.code] = (codes[c.code] || 0) + c.n
    findings.push(F(
      'under-provision-retried',
      'error',
      `${underModelFault.length} complex turn(s) ran on a cheap model and failed ${retries} time(s) for reasons a better model would not have hit`,
      'A cheap model doing expensive work is provable waste when it fails on its own merits: each retry re-sends the whole context, so the downgrade cost more than the premium call it replaced. Retries whose failure code is a provider capacity limit (RATE_LIMIT / SERVER / TIMEOUT / TRANSPORT) are excluded here — those are the provider refusing the load, not the model being wrong, and a premium model would have hit the same wall.',
      { turns: underModelFault.slice(0, 40), retries, codes },
    ))
  }

  if (underCapacity.length > 0) {
    const retries = underCapacity.reduce((a, r) => a + r.retries, 0)
    findings.push(F(
      'under-provision-capacity-retry',
      'info',
      `${underCapacity.length} complex turn(s) on a cheap model were retried ${retries} time(s), but every failure was a provider capacity limit`,
      'These retries are not evidence of a routing mistake: the upstream refused the request (rate limit, server error, timeout), which would have happened on any tier. They are reported because the retries still cost money, but the fix is a provider or quota change, not a routing change.',
      { turns: underCapacity.slice(0, 40), retries },
    ))
  }

  const underClean = under.filter((r) => r.retries === 0)
  if (underClean.length > 0) {
    findings.push(F(
      'under-provision-suspected',
      'info',
      `${underClean.length} complex turn(s) ran on a cheap model without retrying`,
      'These turns succeeded, so there is no proof they were misrouted — but the task shape (long request, many tools, file writes, large context) is what the premium tier exists for. Treat this as a lead to check by hand, not as measured waste.',
      { turns: underClean.slice(0, 40) },
    ))
  }

  // A session where the model changed mid-conversation: the routing was made
  // per-turn by a human, and the switches themselves are the evidence.
  const switches = findSwitches(records)
  if (switches.length > 0) {
    findings.push(F(
      'model-switch',
      'info',
      `${switches.length} model switch(es) inside this session`,
      'The model was changed between turns rather than being routed once. Each switch re-primes the context on a different price point, so a session that hops between tiers pays both models for the same conversation.',
      { switches: switches.slice(0, 30) },
    ))
  }

  return { records, over, under, findings }
}

/** Where the serving model changed from one turn to the next. */
function findSwitches(records) {
  const out = []
  let prev = null
  for (const r of records) {
    if (!r.model) continue
    if (prev && prev.model !== r.model) {
      out.push({ turn: r.turn, from: `${prev.provider}/${prev.model}`, to: `${r.provider}/${r.model}` })
    }
    prev = r
  }
  return out
}

// =====================================================================
// Sweep
// =====================================================================

/**
 * Summarise one session's audit into a row a sweep can rank and total.
 */
export function summariseSession(events, meta = {}) {
  const { records, over, under, findings } = auditSession(events)
  const spend = records.reduce((a, r) => a + (costOf(r) ?? 0), 0)
  const overSpend = over.reduce((a, r) => a + (r.cost ?? 0), 0)
  const overWaste = over.reduce((a, r) => a + (r.estimatedWaste ?? 0), 0)
  const underRetries = under.reduce((a, r) => a + r.retries, 0)
  const retries = records.reduce((a, r) => a + r.retries, 0)
  const errors = findings.filter((f) => f.severity === 'error').length
  const warns = findings.filter((f) => f.severity === 'warn').length
  return {
    name: meta.name ?? null,
    workspace: meta.workspace ?? null,
    file: meta.file ?? null,
    turns: records.length,
    calls: records.reduce((a, r) => a + r.steps, 0),
    inputTokens: records.reduce((a, r) => a + r.inputTokens, 0),
    outputTokens: records.reduce((a, r) => a + r.outputTokens, 0),
    overProvisioned: over.length,
    underProvisioned: under.length,
    underProvisionedRetried: under.filter((r) => r.retries > 0).length,
    retries,
    retriesOnUnderProvisioned: underRetries,
    errors,
    warns,
    estimatedSpend: round4(spend),
    estimatedOverSpend: round4(overSpend),
    estimatedOverWaste: round4(overWaste),
    priced: spend > 0,
    top: findings.find((f) => f.severity === 'error')?.title ?? findings[0]?.title ?? null,
  }
}

function round4(n) {
  return Math.round(n * 10000) / 10000
}

// =====================================================================
// Rules
// =====================================================================

/**
 * Derive routing rules from a set of audited sessions.
 *
 * A rule is only emitted when the evidence supports it, and the support is
 * stated with it — a rule that says "send X to cheap" with no count behind it
 * is a rumour. The shapes are the classification buckets, so a rule reads
 * "turns shaped like this belonged on that tier", not "you should prefer
 * flash", which is the kind of advice that cannot be checked.
 */
export function deriveRules(rows) {
  // rows: per-turn audit rows across sessions
  const byShape = new Map()
  const CAPACITY_CODES = new Set(['RATE_LIMIT', 'TIMEOUT', 'TRANSPORT'])
  for (const r of rows) {
    if (!byShape.has(r.complexity)) byShape.set(r.complexity, { complexity: r.complexity, turns: 0, tokensIn: 0, tokensOut: 0, over: 0, under: 0, underModelFault: 0, retries: 0, models: new Map() })
    const b = byShape.get(r.complexity)
    b.turns += 1
    b.tokensIn += r.inputTokens || 0
    b.tokensOut += r.outputTokens || 0
    b.retries += r.retries || 0
    if (r.tier === 'premium' && (r.complexity === 'trivial' || r.complexity === 'simple')) b.over += 1
    if (r.tier === 'cheap' && r.complexity === 'complex') {
      b.under += 1
      const modelFault = (r.retryCodes || []).some((c) => !CAPACITY_CODES.has(c.code))
      if (modelFault) b.underModelFault += 1
    }
    for (const m of r.models || []) b.models.set(m.model, (b.models.get(m.model) || 0) + m.steps)
  }

  const rules = []
  for (const [complexity, b] of byShape) {
    const recommended = recommendedTier(complexity)
    const models = [...b.models.entries()].sort((a, c) => c[1] - a[1]).map(([m, n]) => `${m}(${n})`)
    const shape = shapeLabel(complexity)
    const rule = {
      id: `route-${complexity}-to-${recommended}`,
      // kept alongside `shape`: the label is for humans, the enum is for code
      complexity,
      shape,
      recommendedTier: recommended,
      evidence: {
        turns: b.turns,
        inputTokens: b.tokensIn,
        outputTokens: b.tokensOut,
        retries: b.retries,
        overProvisioned: b.over,
        underProvisioned: b.under,
        underProvisionedModelFault: b.underModelFault,
        models,
      },
    }
    if (b.over > 0) {
      rule.finding = `${b.over} of ${b.turns} turn(s) shaped ${shape} ran on a premium model — the tier above what the shape needs.`
      rule.saving = 'Route to cheap tier.'
    } else if (b.underModelFault > 0) {
      rule.finding = `${b.underModelFault} of ${b.turns} turn(s) shaped ${shape} ran on a cheap model and failed on its own merits ${b.retries} time(s).`
      rule.saving = 'Route to premium tier; the retries already cost more than the upgrade.'
    } else if (b.under > 0) {
      rule.finding = `${b.under} of ${b.turns} turn(s) shaped ${shape} ran on a cheap model; the retries that followed were provider capacity limits, not model failures.`
      rule.saving = 'Watch this shape — it is premium work that happened to survive on a cheap model.'
    } else {
      rule.finding = `${b.turns} turn(s) shaped ${shape}; no tier mismatch observed.`
      rule.saving = 'Current routing for this shape is fine.'
    }
    rules.push(rule)
  }

  rules.sort((a, b) => {
    const rank = (r) => (r.evidence.overProvisioned > 0 ? 0 : r.evidence.underProvisionedModelFault > 0 ? 1 : r.evidence.underProvisioned > 0 ? 2 : 3)
    return rank(a) - rank(b) || b.evidence.turns - a.evidence.turns
  })
  return rules
}

function shapeLabel(complexity) {
  switch (complexity) {
    case 'trivial': return 'trivial (short ask, no writes)'
    case 'simple': return 'simple (one short instruction, few tools)'
    case 'standard': return 'standard (multi-step, moderate context)'
    default: return 'complex (long brief, many tools, writes, large context)'
  }
}

/** Render rules as the text a human or agent can act on. */
export function renderRules(rules) {
  const lines = ['', `routing rules from ${rules.reduce((a, r) => a + r.evidence.turns, 0)} audited turn(s)`, '']
  for (const r of rules) {
    lines.push(`· ${r.shape}  ->  ${r.recommendedTier} tier`)
    lines.push(`    ${r.finding}`)
    lines.push(`    ${r.saving}`)
    const e = r.evidence
    lines.push(`    evidence: ${e.turns} turn(s), ${e.inputTokens} in / ${e.outputTokens} out tokens, ${e.retries} retry(s)`)
    if (e.models.length) lines.push(`    models seen: ${e.models.slice(0, 5).join(', ')}`)
    lines.push('')
  }
  lines.push('Cost figures anywhere in this report are ESTIMATES from a built-in price table, not invoices.')
  lines.push('')
  return lines.join('\n')
}
