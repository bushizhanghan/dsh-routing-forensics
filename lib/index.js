// dsh-routing-forensics: agent tools over the routing audit pipeline.
//
// Three tools, one pipeline. `routing_audit` replays a single session and says
// which turns were routed to the wrong tier; `routing_sweep` does that for
// every session and totals the money; `routing_rules` turns the same evidence
// into "this shape belongs on that tier". They share the engine so an agent
// asking a routing question gets the same numbers a human gets from the CLI.
//
// Everything is deterministic. No LLM is called and no price is fetched: the
// tier table and price table are declared in the engine, and every money
// figure in every output is labelled an estimate, because the real per-request
// price lives behind the provider's billing API rather than in the session log.
import { join } from 'node:path'
import { homedir } from 'node:os'
import { readdirSync, existsSync } from 'node:fs'
import {
  auditSession,
  deriveRules,
  estimateWaste,
  loadSession,
  replaySession,
  renderRules,
  summariseSession,
  tierOf,
} from './engine.js'

export const name = 'routing-forensics'
export const inject = ['tools']

const Config = {
  // Where DSH keeps session logs.
  sessionsDir: '',
  // Cap the number of sessions a sweep will read, so a large history cannot
  // hang the tool. Override with a bigger number when you want the full sweep.
  maxSessions: 60,
}

const MARK = { error: '✗', warn: '!', info: '·' }

function renderReport(findings, { title }) {
  if (findings.length === 0) return `\n${title}: nothing to report.\n`
  const lines = ['', `${title}: ${findings.length} finding(s)`, '']
  for (const f of findings) {
    lines.push(`${MARK[f.severity] ?? '·'} ${f.title}  [${f.rule}]`)
    lines.push(`    ${f.detail}`)
    if (f.data.turns) {
      for (const t of f.data.turns.slice(0, 8)) {
        const bits = [`turn ${t.turn}`, t.model, t.complexity]
        if (t.retries) bits.push(`${t.retries} retry(s)`)
        if (t.cost != null) bits.push(`$${t.cost.toFixed(4)}`)
        if (t.estimatedWaste != null) bits.push(`waste ~$${t.estimatedWaste.toFixed(4)}`)
        lines.push(`      ${bits.join(' · ')}  score=${t.score} [${(t.reasons || []).join(' ')}]`)
        if (t.userText) lines.push(`        "${t.userText.slice(0, 100).replace(/\n/g, ' ')}"`)
      }
      if (f.data.turns.length > 8) lines.push(`      … ${f.data.turns.length - 8} more`)
    }
    if (f.data.switches) {
      for (const s of f.data.switches.slice(0, 6)) lines.push(`      turn ${s.turn}: ${s.from} -> ${s.to}`)
    }
    if (f.data.estimatedWaste != null) lines.push(`      estimated waste: $${f.data.estimatedWaste.toFixed(4)} (estimate, not an invoice)`)
    if (f.data.retries != null) lines.push(`      retries: ${f.data.retries}`)
    lines.push('')
  }
  const counts = { error: 0, warn: 0, info: 0 }
  for (const f of findings) counts[f.severity] += 1
  const parts = []
  if (counts.error) parts.push(`${counts.error} error`)
  if (counts.warn) parts.push(`${counts.warn} warning`)
  if (counts.info) parts.push(`${counts.info} note`)
  lines.push(parts.join(' · '))
  lines.push('')
  lines.push('All money figures are ESTIMATES from a built-in price table, not invoices.')
  lines.push('')
  return lines.join('\n')
}

/** Resolve the sessions dir: explicit arg, config, or the default per-profile layout. */
function resolveSessionsDir(dir) {
  if (dir) return dir
  if (Config.sessionsDir) return Config.sessionsDir
  const home = homedir()
  return join(home, '.dsh', 'sessions')
}

/**
 * Every session log under a sessions root.
 *
 * The layout is `<sessions>/<workspace-slug>/session-<uuid>/session.v4.jsonl.zstd`
 * — a workspace level sits between the root and the session, so a flat scan of
 * the root finds nothing. Walk down to the session dirs and keep the workspace
 * slug with each row so a report can say which workspace a waste came from.
 */
function listSessions(root, limit) {
  if (!existsSync(root)) return []
  const out = []
  const walk = (dir, workspace, depth) => {
    if (depth > 2 || out.length > (typeof limit === 'number' ? limit : 500)) return
    let entries
    try {
      entries = readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of entries) {
      if (!e.isDirectory()) continue
      const full = join(dir, e.name)
      const log = join(full, 'session.v4.jsonl.zstd')
      if (existsSync(log)) {
        out.push({ name: e.name, workspace, file: log })
        continue
      }
      walk(full, e.name, depth + 1)
    }
  }
  walk(root, '(root)', 0)
  return typeof limit === 'number' ? out.slice(0, limit) : out
}

/** A session path may be given as the .zstd file or as the session directory. */
function resolveSessionFile(p) {
  if (existsSync(p) && p.endsWith('.zstd')) return p
  const f = join(p, 'session.v4.jsonl.zstd')
  return existsSync(f) ? f : null
}

export function apply(ctx, config = {}) {
  const cfg = { ...Config, ...config }
  const OUTPUT_SCHEMA = { type: 'json' }
  const outRender = (_args, value) => [{ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value, null, 2) }]

  const tools = [
    {
      name: 'routing_audit',
      description:
        'Audit the model-routing decisions in ONE session. Replays every model call from the raw event log, classifies each turn\'s task complexity deterministically from the request shape (how much the human wrote, how many tools the turn used, whether it wrote files, whether it was retried, how large the context grew), and reports the two failure modes: over-provision (a premium model doing trivial/simple work — money spent on depth the task never asked for) and under-provision (a cheap model doing complex work, which the log exposes when it had to be retried). Retries whose failure code is a provider capacity limit (RATE_LIMIT/SERVER/TIMEOUT/TRANSPORT) are counted and reported separately, because those are the provider refusing the load rather than the model being wrong. Deterministic, no LLM. Money figures are estimates from a built-in price table, not invoices. Returns {ok, turns, findings, summary, report}.',
      parameters: {
        type: 'object',
        properties: {
          session: { type: 'string', description: 'Path to a session.v4.jsonl.zstd file, or a session directory. Omit to sweep the newest sessions.' },
          dir: { type: 'string', description: 'Sessions root to sweep when no single session is given. Defaults to ~/.dsh/sessions.' },
          limit: { type: 'integer', description: 'How many sessions to sweep at most (default 60).' },
        },
      },
      async execute(args) {
        let file = null
        let label = 'routing audit'
        if (args.session) {
          file = resolveSessionFile(args.session)
          if (!file) return { ok: false, error: `no session.v4.jsonl.zstd at ${args.session}` }
          label = `routing audit · ${file}`
        }
        if (!file) {
          const root = resolveSessionsDir(args.dir)
          const sessions = listSessions(root, Math.max(1, Number(args.limit) || cfg.maxSessions))
          if (sessions.length === 0) return { ok: false, error: `no sessions found under ${root}` }
          file = sessions[0].file
          label = `routing audit · ${sessions[0].workspace}/${sessions[0].name}`
        }
        const { events, error } = loadSession(file)
        if (error) return { ok: false, error }
        if (events.length === 0) return { ok: false, error: 'session log is empty' }
        const result = auditSession(events)
        const records = replaySession(events)
        return {
          ok: true,
          file,
          findings: result.findings,
          turns: records.map((r) => ({
            turn: r.turn,
            model: r.model,
            provider: r.provider,
            tier: r.tier,
            complexity: r.complexity,
            score: r.score,
            reasons: r.reasons,
            shape: r.shape,
            steps: r.steps,
            retries: r.retries,
            retryCodes: r.retryCodes,
            inputTokens: r.inputTokens,
            outputTokens: r.outputTokens,
            estimatedCost: estimateWaste(r).served,
            estimatedWaste: estimateWaste(r).waste,
            userText: r.userText.slice(0, 200),
          })),
          summary: {
            turns: records.length,
            calls: records.reduce((a, r) => a + r.steps, 0),
            overProvisioned: result.over.length,
            underProvisioned: result.under.length,
            underProvisionedRetried: result.under.filter((r) => r.retries > 0).length,
            retries: records.reduce((a, r) => a + r.retries, 0),
            estimatedSpend: round4(records.reduce((a, r) => a + (estimateWaste(r).served ?? 0), 0)),
            estimatedWaste: round4(result.over.reduce((a, r) => a + (r.estimatedWaste ?? 0), 0)),
            estimateNotice: 'money figures are estimates from a built-in price table, not invoices',
          },
          report: renderReport(result.findings, { title: label }),
        }
      },
    },
    {
      name: 'routing_sweep',
      description:
        'Sweep EVERY session under a sessions root and total the routing waste in money, ranked by estimated waste, so you can see which conversations were misrouted worst instead of auditing one at a time. Reports per session: turns, calls, over-provisioned turns, under-provisioned turns, retries, and the estimated spend and estimated waste. Read-only; a big history is capped by `limit` (default 60). Sessions whose models are not in the built-in price table are counted but contribute 0 to the money totals and are reported via the unpriced count. Returns {ok, sessions:[...], totals}.',
      parameters: {
        type: 'object',
        properties: {
          dir: { type: 'string', description: 'Sessions root. Defaults to ~/.dsh/sessions.' },
          limit: { type: 'integer', description: 'Max sessions to read (default 60).' },
        },
      },
      async execute(args) {
        const root = resolveSessionsDir(args.dir)
        const sessions = listSessions(root, Math.max(1, Number(args.limit) || cfg.maxSessions))
        if (sessions.length === 0) return { ok: false, error: `no sessions found under ${root}` }
        const rows = []
        const unpriced = []
        for (const s of sessions) {
          const { events, error } = loadSession(s.file)
          if (error || events.length === 0) continue
          if (!events.some((e) => e.type === 'assistant/message')) continue
          const row = summariseSession(events, s)
          rows.push(row)
          if (!row.priced) unpriced.push(`${s.workspace}/${s.name}`)
        }
        rows.sort((a, b) => b.estimatedOverWaste - a.estimatedOverWaste || b.overProvisioned - a.overProvisioned)
        const totals = rows.reduce(
          (acc, r) => ({
            sessions: acc.sessions + 1,
            turns: acc.turns + r.turns,
            calls: acc.calls + r.calls,
            overProvisioned: acc.overProvisioned + r.overProvisioned,
            underProvisioned: acc.underProvisioned + r.underProvisioned,
            retries: acc.retries + r.retries,
            estimatedSpend: acc.estimatedSpend + r.estimatedSpend,
            estimatedWaste: acc.estimatedWaste + r.estimatedOverWaste,
          }),
          { sessions: 0, turns: 0, calls: 0, overProvisioned: 0, underProvisioned: 0, retries: 0, estimatedSpend: 0, estimatedWaste: 0 },
        )
        for (const k of ['estimatedSpend', 'estimatedWaste']) totals[k] = round4(totals[k])
        // The rules tool renders a report, and the sweep is where those numbers
        // come from, so it needs one too: a caller reading `sessions` raw has to
        // re-derive the story the totals already tell.
        const lines = ['', 'routing waste sweep', '']
        lines.push(`  ${'session'.padEnd(34)} ${'turns'.padStart(6)} ${'over'.padStart(5)} ${'under'.padStart(6)} ${'retry'.padStart(6)} ${'spend'.padStart(9)} ${'waste'.padStart(9)}`)
        for (const r of rows.slice(0, 20)) {
          const label = `${r.workspace ?? '?'}/${r.name ?? '?'}`
          lines.push(
            `  ${String(label).slice(0, 34).padEnd(34)} ${String(r.turns).padStart(6)} ${String(r.overProvisioned).padStart(5)} ${String(r.underProvisioned).padStart(6)} ${String(r.retries).padStart(6)} ${('$' + r.estimatedSpend).padStart(9)} ${('$' + r.estimatedOverWaste).padStart(9)}`,
          )
        }
        lines.push('')
        lines.push(`  ${totals.sessions} session(s), ${totals.turns} turn(s), ${totals.calls} call(s)`)
        lines.push(`  estimated spend $${totals.estimatedSpend}, estimated waste $${totals.estimatedWaste}`)
        if (unpriced.length > 0) lines.push(`  ${unpriced.length} session(s) had models outside the price table — counted, not priced`)
        lines.push('  money figures are estimates from a built-in price table, not invoices')
        lines.push('')
        return {
          ok: true,
          sessions: rows,
          totals,
          unpricedSessions: unpriced,
          report: lines.join('\n'),
          estimateNotice: 'money figures are estimates from a built-in price table, not invoices',
        }
      },
    },
    {
      name: 'routing_rules',
      description:
        'Derive concrete routing rules from the evidence: which task shapes (trivial / simple / standard / complex) belong on which model tier, each rule carrying the count of turns it was derived from, the tokens involved, the retries observed, and the models actually seen serving that shape. A rule is only actionable when the evidence supports it — the over-provisioned and failed-under-provisioned shapes are ranked first, and a shape with no observed mismatch is reported as "current routing is fine" rather than being dressed up as a recommendation. Scope it to one session or sweep a whole history. Returns {ok, rules, report}.',
      parameters: {
        type: 'object',
        properties: {
          session: { type: 'string', description: 'Path to a session.v4.jsonl.zstd file or session directory. Omit to derive rules across every session.' },
          dir: { type: 'string', description: 'Sessions root to sweep when no single session is given. Defaults to ~/.dsh/sessions.' },
          limit: { type: 'integer', description: 'Max sessions to read when sweeping (default 60).' },
        },
      },
      async execute(args) {
        const rows = []
        const collect = (events) => {
          for (const r of replaySession(events)) rows.push(r)
        }
        if (args.session) {
          const file = resolveSessionFile(args.session)
          if (!file) return { ok: false, error: `no session.v4.jsonl.zstd at ${args.session}` }
          const { events, error } = loadSession(file)
          if (error) return { ok: false, error }
          if (events.length === 0) return { ok: false, error: 'session log is empty' }
          collect(events)
        } else {
          const root = resolveSessionsDir(args.dir)
          const sessions = listSessions(root, Math.max(1, Number(args.limit) || cfg.maxSessions))
          if (sessions.length === 0) return { ok: false, error: `no sessions found under ${root}` }
          let used = 0
          for (const s of sessions) {
            const { events, error } = loadSession(s.file)
            if (error || events.length === 0) continue
            if (!events.some((e) => e.type === 'assistant/message')) continue
            collect(events)
            used += 1
          }
          if (used === 0) return { ok: false, error: `no readable sessions under ${root}` }
        }
        if (rows.length === 0) return { ok: false, error: 'no model calls found to derive rules from' }
        const rules = deriveRules(rows)
        return {
          ok: true,
          turns: rows.length,
          rules,
          report: renderRules(rules),
        }
      },
    },
  ]

  for (const t of tools) {
    try {
      ctx.tools.register({
        name: t.name,
        description: t.description,
        parameters: t.parameters,
        output: { schema: OUTPUT_SCHEMA, render: outRender },
        timeoutMs: 300_000,
        isConcurrencySafe: () => true,
        presentCall: (args) => ({ card: 'generic', title: t.name, kind: 'read', rawInput: args }),
        execute: t.execute,
      })
    } catch (error) {
      console.error(`[routing-forensics] ${t.name} registration skipped: ${error}`)
    }
  }
}

function round4(n) {
  return Math.round(n * 10000) / 10000
}
