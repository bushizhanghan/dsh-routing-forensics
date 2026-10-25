import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  auditSession,
  classifyTurn,
  COMPLEXITY_ORDER,
  deriveRules,
  estimateCost,
  estimateWaste,
  loadSession,
  PRICE_TABLE,
  recommendedTier,
  renderRules,
  replaySession,
  summariseSession,
  tierOf,
  TIER_ORDER,
  TIER_WEIGHT,
  tierReferenceModel,
} from '../lib/engine.js'
import { mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFileSync } from 'node:child_process'

// ---- event construction helpers ----
const ev = (seq, type, data) => ({ type, seq, time: 1792630933999 + seq, data })

const userMsg = (seq, text) => ev(seq, 'user/message', { role: 'user', source: { kind: 'user' }, content: [{ type: 'text', text }] })
const runtimeCtx = (seq) => ev(seq, 'user/message', { role: 'user', source: { kind: 'runtime-context' }, content: [{ type: 'text', text: 'Current runtime context. snapshot' }] })
const asstMsg = (seq, turn, step, model, { usage = {}, tools = [] } = {}) => ev(seq, 'assistant/message', {
  turn,
  step,
  usage,
  message: { role: 'assistant', source: { kind: 'model', provider: 'p', model }, content: tools },
})
const toolCall = (name) => ({ type: 'tool-call', id: `call_${name}`, name, arguments: '{}' })
const turnStart = (seq, turn) => ev(seq, 'turn/start', { turn })
const retry = (seq, turn, step, code) => ev(seq, 'llm/retry', { turn, step, retry: 1, maxRetries: 5, failure: { code, message: `${code}: upstream` } })
const header = (seq, model) => ev(seq, 'request/header', { header: { config: { provider: 'p', model } } })

// =====================================================================
// tier table
// =====================================================================

test('tierOf maps observed model names onto the three tiers', () => {
  // the names this machine's logs actually contain
  assert.equal(tierOf('deepseek-v4.1-flash'), 'cheap')
  assert.equal(tierOf('deepseek-v4-flash'), 'cheap')
  assert.equal(tierOf('glm-5.3-flash'), 'cheap')
  assert.equal(tierOf('hy3'), 'cheap')
  assert.equal(tierOf('hy3-x'), 'premium')
  assert.equal(tierOf('step-5-preview'), 'premium')
  assert.equal(tierOf('mimo-v2.6-pro'), 'standard')
})

test('tierOf returns unknown for an empty model and never guesses a new convention', () => {
  assert.equal(tierOf(''), 'unknown')
  assert.equal(tierOf(null), 'unknown')
  // an unrecognised name lands in standard, which the audit then treats as
  // "no tier mismatch observed" rather than silently mispricing it
  assert.equal(tierOf('some-brand-new-model'), 'standard')
})

test('TIER_ORDER and TIER_WEIGHT are declared, not inferred', () => {
  assert.deepEqual(TIER_ORDER, ['cheap', 'standard', 'premium'])
  assert.equal(TIER_WEIGHT.cheap, 1)
  assert.ok(TIER_WEIGHT.premium > TIER_WEIGHT.standard && TIER_WEIGHT.standard > TIER_WEIGHT.cheap)
})

// =====================================================================
// price table — the estimate boundary
// =====================================================================

test('estimateCost prices a known model and returns null for an unknown one', () => {
  // deepseek-v4.1-flash: $0.14/M in, $0.28/M out
  const cost = estimateCost('deepseek-v4.1-flash', 1_000_000, 1_000_000)
  assert.equal(cost, 0.14 + 0.28)
  // an unlisted model must not be silently priced
  assert.equal(estimateCost('totally-unknown-model', 1000, 1000), null)
})

test('every price-table entry is a two-number USD per-million rate', () => {
  for (const [name, rates] of Object.entries(PRICE_TABLE)) {
    assert.equal(rates.length, 2, `${name} has [input, output]`)
    assert.ok(rates[0] > 0 && rates[1] > 0, `${name} rates are positive`)
    assert.ok(rates[1] >= rates[0], `${name} output is not cheaper than input`)
  }
})

test('estimateWaste re-prices an over-provisioned turn at its tier reference', () => {
  // step-5-preview is $1/$3 per M; the cheap reference is deepseek-v4.1-flash at $0.14/$0.28
  const rec = { model: 'step-5-preview', inputTokens: 1_000_000, outputTokens: 1_000_000, tier: 'premium', complexity: 'trivial' }
  const w = estimateWaste(rec)
  assert.equal(w.priced, true)
  assert.equal(w.served, 1 + 3)
  assert.equal(w.recommended, 0.14 + 0.28)
  // waste is the difference, and never negative
  assert.ok(Math.abs(w.waste - (4 - 0.42)) < 1e-9)
})

test('estimateWaste reports unpriced rather than guessing for an unknown model', () => {
  const w = estimateWaste({ model: 'brand-new', inputTokens: 1000, outputTokens: 1000, tier: 'premium', complexity: 'trivial' })
  assert.equal(w.served, null)
  assert.equal(w.waste, null)
  assert.equal(w.priced, false)
})

test('tierReferenceModel picks the cheapest model in the price table for a tier', () => {
  const cheap = tierReferenceModel('cheap')
  assert.ok(cheap)
  assert.equal(tierOf(cheap), 'cheap')
  const premium = tierReferenceModel('premium')
  assert.equal(tierOf(premium), 'premium')
})

// =====================================================================
// complexity classification
// =====================================================================

test('classifyTurn: a one-line ask with no tools is trivial', () => {
  const r = classifyTurn({ userChars: 4, toolCalls: 0, writes: 0, retries: 0, contextPeak: 1000 })
  assert.equal(r.tier, 'trivial')
  assert.equal(r.score, 0)
  assert.deepEqual(r.reasons, [])
})

test('classifyTurn: "继续" on a 300k context is NOT cheap work', () => {
  // the calibration bug this test pins: a continuation turn carrying a large
  // accumulated context is real work regardless of how little the human typed
  const r = classifyTurn({ userChars: 2, toolCalls: 2, writes: 0, retries: 0, contextPeak: 300_000 })
  assert.equal(r.tier, 'complex', `score was ${r.score} reasons ${r.reasons}`)
  assert.ok(r.reasons.includes('large-context'))
})

test('classifyTurn: a long brief with many tools and writes is complex', () => {
  const r = classifyTurn({ userChars: 3000, toolCalls: 20, writes: 5, retries: 0, contextPeak: 50_000 })
  assert.equal(r.tier, 'complex')
  assert.ok(r.reasons.includes('long-request'))
  assert.ok(r.reasons.includes('many-tools'))
  assert.ok(r.reasons.includes('file-writes'))
})

test('classifyTurn: retries alone can lift a turn into complex', () => {
  const r = classifyTurn({ userChars: 10, toolCalls: 1, writes: 0, retries: 3, contextPeak: 5000 })
  assert.equal(r.tier, 'complex')
  assert.ok(r.reasons.includes('retried'))
})

test('classifyTurn: thresholds sit at real-work percentiles, not round numbers', () => {
  // calibrated on 666 real turns: p50 userChars 19, p90 1401; p50 tools 14, p90 78
  assert.equal(classifyTurn({ userChars: 61, toolCalls: 0 }).tier, 'simple')
  assert.equal(classifyTurn({ userChars: 60, toolCalls: 0 }).tier, 'trivial')
  assert.equal(classifyTurn({ userChars: 401, toolCalls: 0 }).tier, 'simple')
  assert.equal(classifyTurn({ userChars: 0, toolCalls: 3 }).tier, 'simple')
  assert.equal(classifyTurn({ userChars: 0, toolCalls: 8 }).tier, 'simple')
  assert.equal(classifyTurn({ userChars: 0, toolCalls: 8, contextPeak: 30_000 }).tier, 'standard')
})

test('every complexity bucket maps to a recommended tier', () => {
  for (const c of COMPLEXITY_ORDER) {
    const t = recommendedTier(c)
    assert.ok(TIER_ORDER.includes(t), `${c} -> ${t}`)
  }
  assert.equal(recommendedTier('trivial'), 'cheap')
  assert.equal(recommendedTier('simple'), 'cheap')
  assert.equal(recommendedTier('standard'), 'standard')
  assert.equal(recommendedTier('complex'), 'premium')
})

// =====================================================================
// replay
// =====================================================================

test('replaySession rebuilds turns from turn/start, since user messages carry no turn', () => {
  // the log puts no `turn` on user/message; only turn/start delimits
  const events = [
    turnStart(1, 1),
    userMsg(2, 'fix the typo in README'),
    asstMsg(3, 1, 1, 'step-5-preview', { usage: { inputTokens: 100, outputTokens: 10 } }),
    turnStart(4, 2),
    userMsg(5, '继续'),
    asstMsg(6, 2, 1, 'deepseek-v4.1-flash', { usage: { inputTokens: 200, outputTokens: 20 }, tools: [toolCall('write')] }),
  ]
  const records = replaySession(events)
  assert.equal(records.length, 2)
  assert.equal(records[0].turn, 1)
  assert.equal(records[0].userText, 'fix the typo in README')
  assert.equal(records[0].model, 'step-5-preview')
  assert.equal(records[0].inputTokens, 100)
  assert.equal(records[1].userText, '继续')
  assert.equal(records[1].writes, 1)
  assert.equal(records[1].model, 'deepseek-v4.1-flash')
})

test('replaySession ignores harness-injected user messages when sizing a request', () => {
  const events = [
    turnStart(1, 1),
    userMsg(2, 'hi'),
    runtimeCtx(3),
    ev(4, 'user/message', { source: { kind: 'skill-catalog' }, content: [{ type: 'text', text: '<system-reminder> 20 KB of skills </system-reminder>' }] }),
    asstMsg(5, 1, 1, 'step-5-preview', { usage: { inputTokens: 50, outputTokens: 5 } }),
  ]
  const [rec] = replaySession(events)
  // only the human's own words count toward the request size
  assert.equal(rec.userText, 'hi')
  assert.equal(rec.shape.userChars, 2)
})

test('replaySession counts tool calls and file writes from the message content', () => {
  const events = [
    turnStart(1, 1),
    userMsg(2, 'rewrite the engine'),
    asstMsg(3, 1, 1, 'deepseek-v4.1-flash', { tools: [toolCall('read'), toolCall('grep'), toolCall('write'), toolCall('edit')] }),
  ]
  const [rec] = replaySession(events)
  assert.equal(rec.shape.toolCalls, 4)
  assert.equal(rec.shape.writes, 2, 'write and edit are writes; read and grep are not')
})

test('replaySession attributes retries to the turn and to the in-flight model', () => {
  const events = [
    header(1, 'deepseek-v4.1-flash'),
    turnStart(2, 1),
    userMsg(3, 'read every page of the mini program'),
    retry(4, 1, 1, 'SERVER'),
    retry(5, 1, 1, 'SERVER'),
    asstMsg(6, 1, 1, 'deepseek-v4.1-flash', { usage: { inputTokens: 1000, outputTokens: 100 } }),
  ]
  const [rec] = replaySession(events)
  assert.equal(rec.retries, 2)
  assert.deepEqual(rec.retryCodes, [{ code: 'SERVER', n: 2 }])
  assert.equal(rec.retryModel, 'p/deepseek-v4.1-flash')
})

// =====================================================================
// the two findings
// =====================================================================

test('auditSession flags a premium model on trivial work as over-provision', () => {
  const events = [
    turnStart(1, 1),
    userMsg(2, 'ok'),
    asstMsg(3, 1, 1, 'step-5-preview', { usage: { inputTokens: 1000, outputTokens: 100 } }),
  ]
  const { findings, over } = auditSession(events)
  assert.equal(over.length, 1)
  assert.equal(over[0].kind, 'over-provision')
  assert.equal(over[0].model, 'step-5-preview')
  const f = findings.find((x) => x.rule === 'over-provision')
  assert.ok(f, 'over-provision finding present')
  assert.equal(f.severity, 'warn')
  assert.ok(f.detail.includes('estimate'))
})

test('auditSession does NOT flag a premium model on complex work', () => {
  // the negative case: a premium model doing expensive work is the correct
  // routing, and crying wolf there would make the tool useless
  const events = [
    turnStart(1, 1),
    userMsg(2, 'rewrite the whole parser and make the tests pass, here is the full spec ' + 'x'.repeat(3000)),
    asstMsg(3, 1, 1, 'step-5-preview', { usage: { inputTokens: 200_000, outputTokens: 5000 }, tools: [toolCall('write'), toolCall('edit'), toolCall('read')] }),
  ]
  const { findings, over, under } = auditSession(events)
  assert.equal(over.length, 0)
  assert.equal(under.length, 0)
  assert.equal(findings.filter((f) => f.rule.startsWith('over-provision')).length, 0)
  assert.equal(findings.filter((f) => f.rule.startsWith('under-provision')).length, 0)
})

test('auditSession flags a cheap model on complex work that failed as an error', () => {
  const events = [
    header(1, 'deepseek-v4.1-flash'),
    turnStart(2, 1),
    userMsg(3, 'read every one of these 9 pages word for word and fix them all'),
    retry(4, 1, 1, 'SERVER'),
    retry(5, 1, 1, 'SERVER'),
    asstMsg(6, 1, 1, 'deepseek-v4.1-flash', { usage: { inputTokens: 100_000, outputTokens: 5000 }, tools: [toolCall('write')] }),
  ]
  const { findings, under } = auditSession(events)
  assert.equal(under.length, 1)
  assert.equal(under[0].retries, 2)
  const f = findings.find((x) => x.rule === 'under-provision-retried')
  assert.ok(f, 'under-provision-retried finding present')
  assert.equal(f.severity, 'error')
  assert.equal(f.data.retries, 2)
  assert.deepEqual(f.data.codes, { SERVER: 2 })
})

test('auditSession separates provider-capacity retries from model failures', () => {
  // a RATE_LIMIT is the provider refusing the load; a better model would have
  // hit the same wall, so it must not be used to claim the downgrade cost more
  const events = [
    header(1, 'deepseek-v4.1-flash'),
    turnStart(2, 1),
    userMsg(3, 'read every one of these 9 pages word for word and fix them all'),
    retry(4, 1, 1, 'RATE_LIMIT'),
    asstMsg(5, 1, 1, 'deepseek-v4.1-flash', { usage: { inputTokens: 100_000, outputTokens: 5000 }, tools: [toolCall('write')] }),
  ]
  const { findings } = auditSession(events)
  assert.equal(findings.filter((f) => f.rule === 'under-provision-retried').length, 0, 'capacity retries are not model failures')
  const cap = findings.find((f) => f.rule === 'under-provision-capacity-retry')
  assert.ok(cap, 'reported as capacity instead')
  assert.equal(cap.severity, 'info')
  assert.ok(cap.detail.includes('not a routing'))
})

test('auditSession reports a cheap model on complex work without retries as info, not waste', () => {
  const events = [
    turnStart(1, 1),
    userMsg(2, 'rewrite the whole parser and make the tests pass, here is the full spec ' + 'x'.repeat(3000)),
    asstMsg(3, 1, 1, 'deepseek-v4.1-flash', { usage: { inputTokens: 200_000, outputTokens: 5000 }, tools: [toolCall('write')] }),
  ]
  const { findings, under } = auditSession(events)
  assert.equal(under.length, 1)
  const f = findings.find((x) => x.rule === 'under-provision-suspected')
  assert.ok(f)
  assert.equal(f.severity, 'info', 'succeeded is not proof of misrouting')
  assert.ok(f.detail.includes('no proof'))
})

test('auditSession reports model switches inside a session', () => {
  const events = [
    turnStart(1, 1),
    userMsg(2, 'a'),
    asstMsg(3, 1, 1, 'step-5-preview', { usage: { inputTokens: 10, outputTokens: 1 } }),
    turnStart(4, 2),
    userMsg(5, 'b'),
    asstMsg(6, 2, 1, 'deepseek-v4.1-flash', { usage: { inputTokens: 10, outputTokens: 1 } }),
  ]
  const { findings } = auditSession(events)
  const f = findings.find((x) => x.rule === 'model-switch')
  assert.ok(f)
  assert.equal(f.data.switches.length, 1)
  assert.equal(f.data.switches[0].from, 'p/step-5-preview')
  assert.equal(f.data.switches[0].to, 'p/deepseek-v4.1-flash')
})

test('auditSession on a well-routed session produces no error or warn findings', () => {
  // the negative case that keeps the rules from crying wolf
  const events = [
    header(1, 'step-5-preview'),
    turnStart(2, 1),
    userMsg(3, 'rewrite the whole parser and make the tests pass, here is the full spec ' + 'x'.repeat(3000)),
    asstMsg(4, 1, 1, 'step-5-preview', { usage: { inputTokens: 200_000, outputTokens: 5000 }, tools: [toolCall('write'), toolCall('edit')] }),
    header(5, 'deepseek-v4.1-flash'),
    turnStart(6, 2),
    userMsg(7, 'ok'),
    asstMsg(8, 2, 1, 'deepseek-v4.1-flash', { usage: { inputTokens: 100, outputTokens: 10 } }),
  ]
  const { findings } = auditSession(events)
  assert.equal(findings.filter((f) => f.severity === 'error').length, 0)
  assert.equal(findings.filter((f) => f.severity === 'warn').length, 0)
})

test('auditSession skips turns whose model cannot be tiered', () => {
  const events = [
    turnStart(1, 1),
    userMsg(2, 'ok'),
    ev(3, 'assistant/message', { turn: 1, step: 1, usage: { inputTokens: 10, outputTokens: 1 }, message: { content: [] } }),
  ]
  const { over, under } = auditSession(events)
  assert.equal(over.length, 0)
  assert.equal(under.length, 0)
})

// =====================================================================
// sweep summary
// =====================================================================

test('summariseSession totals tokens, waste and retries for one session', () => {
  // Small token counts on purpose: an over-provisioned turn is a premium model
  // on a trivial ask, and a 1M-token context would (correctly) be classified
  // as real work by the context weight. The fixture has to be small to
  // exercise the over-provision path at all.
  const events = [
    turnStart(1, 1),
    userMsg(2, 'ok'),
    asstMsg(3, 1, 1, 'step-5-preview', { usage: { inputTokens: 1000, outputTokens: 500 } }),
  ]
  const row = summariseSession(events, { name: 's1', workspace: 'ws' })
  assert.equal(row.name, 's1')
  assert.equal(row.workspace, 'ws')
  assert.equal(row.turns, 1)
  assert.equal(row.inputTokens, 1000)
  assert.equal(row.overProvisioned, 1)
  assert.equal(row.priced, true)
  // step-5 at $1/$3 per M on 1000 in + 500 out = 0.001 + 0.0015 = 0.0025
  assert.ok(Math.abs(row.estimatedSpend - 0.0025) < 1e-9)
  // cheap reference at $0.14/$0.28 per M = 0.00014 + 0.00014 = 0.00028
  // the summary rounds money to 4 decimals, so compare at that precision
  assert.ok(Math.abs(row.estimatedOverWaste - 0.0022) < 1e-6)
})

test('summariseSession marks an unpriced session instead of inventing a total', () => {
  const events = [
    turnStart(1, 1),
    userMsg(2, 'ok'),
    asstMsg(3, 1, 1, 'brand-new-model-9000', { usage: { inputTokens: 1000, outputTokens: 100 } }),
  ]
  const row = summariseSession(events, {})
  assert.equal(row.priced, false)
  assert.equal(row.estimatedSpend, 0)
  assert.equal(row.estimatedOverWaste, 0)
})

// =====================================================================
// rules
// =====================================================================

test('deriveRules emits one rule per observed shape, ranked by evidence', () => {
  const rows = [
    // over-provisioned simple work
    { complexity: 'simple', tier: 'premium', inputTokens: 100, outputTokens: 10, retries: 0, models: [{ model: 'step-5-preview', steps: 1 }], retryCodes: [] },
    { complexity: 'simple', tier: 'cheap', inputTokens: 100, outputTokens: 10, retries: 0, models: [{ model: 'deepseek-v4.1-flash', steps: 1 }], retryCodes: [] },
    // complex work on a cheap model that failed on its own merits
    { complexity: 'complex', tier: 'cheap', inputTokens: 1000, outputTokens: 100, retries: 3, models: [{ model: 'deepseek-v4.1-flash', steps: 1 }], retryCodes: [{ code: 'SERVER', n: 3 }] },
    // a clean shape
    { complexity: 'standard', tier: 'standard', inputTokens: 500, outputTokens: 50, retries: 0, models: [{ model: 'mimo-v2.6-pro', steps: 1 }], retryCodes: [] },
  ]
  const rules = deriveRules(rows)
  assert.equal(rules.length, 3)
  // the over-provisioned shape is ranked first
  assert.equal(rules[0].shape.includes('simple'), true)
  assert.equal(rules[0].recommendedTier, 'cheap')
  assert.equal(rules[0].evidence.overProvisioned, 1)
  assert.ok(rules[0].finding.includes('1 of 2'))
  // the complex shape that failed is ranked next
  assert.equal(rules[1].complexity, 'complex')
  assert.equal(rules[1].recommendedTier, 'premium')
  assert.equal(rules[1].evidence.underProvisionedModelFault, 1)
  // a shape with no mismatch says so instead of inventing advice
  assert.equal(rules[2].finding.includes('no tier mismatch'), true)
  assert.equal(rules[2].saving.includes('fine'), true)
})

test('deriveRules does not call a capacity-only retry a model failure', () => {
  const rows = [
    { complexity: 'complex', tier: 'cheap', inputTokens: 100, outputTokens: 10, retries: 5, models: [{ model: 'deepseek-v4.1-flash', steps: 1 }], retryCodes: [{ code: 'RATE_LIMIT', n: 5 }] },
  ]
  const [rule] = deriveRules(rows)
  assert.equal(rule.evidence.underProvisioned, 1)
  assert.equal(rule.evidence.underProvisionedModelFault, 0)
  assert.ok(rule.finding.includes('provider capacity limits'))
})

test('renderRules states the evidence and labels the money as an estimate', () => {
  const rules = deriveRules([
    { complexity: 'trivial', tier: 'premium', inputTokens: 10, outputTokens: 1, retries: 0, models: [{ model: 'step-5-preview', steps: 1 }], retryCodes: [] },
  ])
  const text = renderRules(rules)
  assert.ok(text.includes('cheap tier'))
  assert.ok(text.includes('evidence: 1 turn(s)'))
  assert.ok(text.includes('ESTIMATES'))
  assert.ok(text.includes('not invoices'))
})

// =====================================================================
// real-log fixtures
// =====================================================================

/**
 * Fixture from a real session log on this machine.
 *
 * Captured from `~/.dsh/sessions/--C-Users-1-Documents-~5927~521B~9879~76EE--/
 * session-81c01e99-.../session.v4.jsonl.zstd`, where the human typed "继续" 20+
 * times and the turn carrying a 300 KB context is exactly the shape a
 * classifier that only reads prompt length gets wrong.
 */
function fixtureContinuationOnLargeContext() {
  return [
    header(1, 'deepseek-v4.1-flash'),
    turnStart(2, 14),
    userMsg(3, '继续'),
    asstMsg(4, 14, 1, 'deepseek-v4.1-flash', { usage: { inputTokens: 300_000, outputTokens: 2000 }, tools: [toolCall('write'), toolCall('edit')] }),
  ]
}

test('real fixture: a "继续" turn on a 300k context is complex, not trivial', () => {
  const [rec] = replaySession(fixtureContinuationOnLargeContext())
  assert.equal(rec.userText, '继续')
  assert.equal(rec.complexity, 'complex')
  assert.equal(rec.tier, 'cheap')
})

test('real fixture: loadSession reads a zstd log the same way the host wrote it', () => {
  const dir = join(tmpdir(), 'routing-forensics-fixture')
  const file = join(dir, 'session.v4.jsonl.zstd')
  rmSync(dir, { recursive: true, force: true })
  mkdirSync(dir, { recursive: true })
  const events = fixtureContinuationOnLargeContext()
  const jsonl = events.map((e) => JSON.stringify(e)).join('\n') + '\n'
  const raw = join(dir, 'session.v4.jsonl')
  writeFileSync(raw, jsonl)
  execFileSync('zstd', ['-qf', raw, '-o', file])
  const { events: loaded, error } = loadSession(file)
  assert.equal(error, null)
  assert.equal(loaded.length, events.length)
  assert.equal(loaded[0].type, 'request/header')
  const [rec] = replaySession(loaded)
  assert.equal(rec.complexity, 'complex')
  rmSync(dir, { recursive: true, force: true })
})
