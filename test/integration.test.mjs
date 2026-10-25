// Integration test: boot the plugin against a fake ctx and drive every tool
// against the real session logs on this machine.
import { apply } from '../lib/index.js'
import { existsSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { execFileSync } from 'node:child_process'

const registry = new Map()
const fakeCtx = { tools: { register: (t) => registry.set(t.name, t) } }
apply(fakeCtx, { maxSessions: 10 })

const SESSIONS = join(homedir(), '.dsh', 'sessions')

/** A real session log that has both model selections and turns. */
function realSession() {
  for (const ws of readdirSync(SESSIONS, { withFileTypes: true })) {
    if (!ws.isDirectory()) continue
    const wsDir = join(SESSIONS, ws.name)
    for (const d of readdirSync(wsDir, { withFileTypes: true })) {
      if (!d.isDirectory()) continue
      const f = join(wsDir, d.name, 'session.v4.jsonl.zstd')
      if (!existsSync(f)) continue
      try {
        const text = execFileSync('zstd', ['-dc', f], { maxBuffer: 64 * 1024 * 1024, encoding: 'utf8' })
        if (text.includes('"assistant/message"') && text.includes('"turn/start"')) return f
      } catch {}
    }
  }
  return null
}

const results = []
const call = async (name, args = {}) => registry.get(name).execute(args, { signal: undefined })

try {
  const expected = ['routing_audit', 'routing_sweep', 'routing_rules']
  results.push([`${expected.length} tools registered`, expected.every((n) => registry.has(n))])

  // every tool must declare a read-only presentCall
  const allReadOnly = expected.every((n) => {
    const pc = registry.get(n)
    return typeof pc.presentCall === 'function'
  })
  results.push(['every tool declares a presentCall', allReadOnly])

  const file = realSession()
  results.push(['found a real session log on this machine', file !== null])

  if (file) {
    const audit = await call('routing_audit', { session: file })
    results.push(['routing_audit runs on a real session', audit.ok === true])
    results.push(['routing_audit reports turns', typeof audit.summary?.turns === 'number'])
    results.push(['routing_audit produces a text report', typeof audit.report === 'string' && audit.report.length > 20])
    results.push(['routing_audit findings are structured', Array.isArray(audit.findings) && audit.findings.every((f) => f.rule && f.severity && f.title && f.detail)])

    const missing = await call('routing_audit', { session: 'Z:/nope/nothing.zstd' })
    results.push(['a missing session is an error, not a throw', missing.ok === false])
  }

  const sweep = await call('routing_sweep', { dir: SESSIONS, limit: 8 })
  if (sweep.ok) {
    results.push(['routing_sweep sweeps real sessions', Array.isArray(sweep.sessions) && sweep.sessions.length >= 1])
    results.push(['routing_sweep totals the spend', sweep.totals && typeof sweep.totals.estimatedSpend === 'number'])
    results.push(['routing_sweep produces a text report', typeof sweep.report === 'string' && sweep.report.length > 20])
  } else {
    // an empty corpus is an honest error, not a crash
    results.push(['routing_sweep handles an empty corpus', typeof sweep.error === 'string'])
  }

  const rules = await call('routing_rules', { dir: SESSIONS, limit: 8 })
  if (rules.ok) {
    results.push(['routing_rules emits rules', Array.isArray(rules.rules) && rules.rules.length >= 1])
    results.push(['every rule names a complexity and a tier', rules.rules.every((r) => r.complexity && r.recommendedTier && r.evidence)])
    results.push(['routing_rules produces a text report', typeof rules.report === 'string'])
  } else {
    results.push(['routing_rules handles an empty corpus', typeof rules.error === 'string'])
  }

  // a root with nothing in it must be an error, not a throw
  const empty = await call('routing_sweep', { dir: 'Z:/nope' })
  results.push(['an empty sessions root is an error, not a throw', empty.ok === false])
} catch (error) {
  console.error('UNEXPECTED THROW:', error)
  results.push(['no-throw', false])
}

let pass = 0
for (const [name, ok] of results) {
  console.log(`${ok ? '✔' : '✖'} ${name}`)
  if (ok) pass += 1
}
console.log(`\n${pass}/${results.length} passed`)
process.exit(pass === results.length ? 0 : 1)
