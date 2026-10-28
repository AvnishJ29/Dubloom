import test from 'node:test'
import assert from 'node:assert/strict'
import { paces, fitLine } from '../src/lib/timing.js'

// one voice at 10 characters per second: a 4 s slot holds about 40 characters
const lines = [
  { start: 0, end: 4, speaker: 'A', voiced_text: 'x'.repeat(40), fit_ratio: 1.0 },
  { start: 5, end: 9, speaker: 'A', voiced_text: 'x'.repeat(40), fit_ratio: 1.0 },
]
const rates = paces(lines)

test('the pace comes from the voiced clip', () => assert.equal(Math.round(rates[0]), 10))
test('a line within its slot fits; borrowing the pause after it still fits', () => {
  assert.equal(fitLine(lines, rates, 0, 'x'.repeat(44)).kind, 'ok')
  assert.match(fitLine(lines, rates, 0, 'x'.repeat(50)).text, /pause after it/) // 5 s: room is 4.85 s at 1.15x
})
test('a rushed line warns and an overlong line says its end is cut, with how much to remove', () => {
  assert.equal(fitLine(lines, rates, 0, 'x'.repeat(60)).kind, 'warn')
  const cut = fitLine(lines, rates, 0, 'x'.repeat(80))
  assert.equal(cut.kind, 'bad')
  assert.match(cut.text, /Remove about 25 characters/) // 80 - 4.85 s x 1.15 x 10
})
test('no pace, no meter', () =>
  assert.equal(
    fitLine(
      [{ start: 0, end: 2, speaker: 'B' }],
      paces([{ start: 0, end: 2, speaker: 'B' }]),
      0,
      'abc',
    ),
    null,
  ))

import { timingIssue } from '../src/lib/timing.js'
test('cut and rushed lines are flagged; a borrowed pause or a natural pace is not', () => {
  assert.equal(
    timingIssue({ timing_trimmed: true, timing_ratio: 1.3, fit_ratio: 2.64 }).kind,
    'bad',
  )
  assert.match(
    timingIssue({ timing_trimmed: true, timing_ratio: 1.3, fit_ratio: 2.64 }).text,
    /2\.6x/,
  )
  assert.equal(timingIssue({ timing_ratio: 1.3 }).kind, 'bad') // older renders: the ceiling means cut
  assert.equal(timingIssue({ timing_ratio: 1.22 }).kind, 'warn')
  assert.equal(timingIssue({ timing_ratio: 1.15, timing_borrowed: 0.6 }), null)
  assert.equal(timingIssue({ timing_ratio: 0.9 }), null)
})

import { nextStarts } from '../src/lib/timing.js'
test('the recorded trimmed flag wins over the speed heuristic; the next start is found once per script', () => {
  assert.equal(timingIssue({ timing_trimmed: false, timing_ratio: 1.3 }).kind, 'warn') // natural mode: fast, not cut
  assert.deepEqual(nextStarts([{ start: 5 }, { start: 0 }, { start: 9 }]), [9, 5, Infinity])
})
