import test from 'node:test'
import assert from 'node:assert/strict'
import { parseClock, validVideoUrl } from '../src/lib/validation.js'
import { subscribe, request } from '../src/lib/api.js'

test('clip times accept seconds and clocks, rejecting malformed or negative inputs', () => {
  for (const [input, expected] of [
    ['90', 90],
    ['1:30', 90],
    ['1:02:03.5', 3723.5],
    ['0', 0],
  ])
    assert.equal(parseClock(input), expected)
  for (const input of ['', '-1', '1:99', '1:2:3:4', 'hello', 'Infinity', '1::2'])
    assert.ok(Number.isNaN(parseClock(input)), input)
})
test('only HTTP video URLs are accepted', () => {
  assert.ok(validVideoUrl('https://example.com/video.mp4'))
  for (const value of ['garbage input', 'file:///tmp/a.mp4', 'javascript:alert(1)'])
    assert.equal(validVideoUrl(value), false)
})
test('API validation failures preserve field feedback', async () => {
  const original = globalThis.fetch
  globalThis.fetch = async () => ({
    ok: false,
    status: 422,
    json: async () => ({ detail: [{ loc: ['body', 'language'], msg: 'Required' }] }),
  })
  try {
    await assert.rejects(request('/jobs'), /language: Required/)
  } finally {
    globalThis.fetch = original
  }
})
test('late responses cannot update an unsubscribed project', async () => {
  const original = globalThis.fetch
  let respond,
    updates = 0
  globalThis.fetch = () =>
    new Promise((resolve) => {
      respond = resolve
    })
  const stop = subscribe('old', { onData: () => updates++ })
  stop()
  respond({ ok: true, json: async () => ({ status: 'running' }) })
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(updates, 0)
  globalThis.fetch = original
})
test('terminal job closes the stream and completes once', async () => {
  const originalFetch = globalThis.fetch,
    originalSource = globalThis.EventSource
  let stream,
    done = 0,
    closes = 0
  globalThis.EventSource = class {
    constructor() {
      stream = this
    }
    addEventListener() {}
    close() {
      closes++
    }
  }
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ status: 'completed' }) })
  const stop = subscribe('finished', { onData: () => {}, onDone: () => done++ })
  await new Promise((resolve) => setImmediate(resolve))
  stream.onmessage({ data: JSON.stringify({ status: 'completed' }) })
  assert.equal(done, 1)
  assert.equal(closes, 1)
  stop()
  globalThis.fetch = originalFetch
  globalThis.EventSource = originalSource
})
test('expired login is reported by job polling', async () => {
  const original = globalThis.fetch
  let status
  globalThis.fetch = async () => ({
    ok: false,
    status: 401,
    json: async () => ({ detail: 'Sign in' }),
  })
  const stop = subscribe('private', {
    onData: () => {},
    onError: (error) => {
      status = error.status
    },
  })
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(status, 401)
  stop()
  globalThis.fetch = original
})

test("history lists only the owner's changes; a studio re-render just refreshes the entry before it", async () => {
  const { generations } = await import('../src/lib/history.js')
  const w1l4 = {
    status: 'completed',
    video_url: '/v',
    finished_at: 'd',
    versions: [
      { number: 1, rendered_at: 'a', reason: 'Script edited: 7 lines' },
      { number: 2, rendered_at: 'b', reason: 'Re-rendered' },
      { number: 3, rendered_at: 'c', reason: 'Script edited: 8 lines' },
    ],
  }
  assert.deepEqual(
    generations(w1l4).map((r) => [r.name, r.number, r.made]),
    [
      ['Latest', 0, 'Script edited: 8 lines'],
      ['Version 2', 3, 'Script edited: 7 lines'],
      ['Version 1', 1, 'First dub'],
    ],
  )
  const refreshedOnly = {
    status: 'completed',
    video_url: '/v',
    versions: [
      { number: 1, reason: 'Re-rendered' },
      { number: 2, reason: 'Improved lip sync and timing' },
    ],
  }
  assert.deepEqual(
    generations(refreshedOnly).map((r) => [r.name, r.number, r.made]),
    [['Latest', 0, 'First dub']],
  )
  assert.deepEqual(
    generations({
      status: 'failed',
      versions: [{ number: 1, reason: 'Script edited: 1 line' }],
    }).map((r) => r.name),
    ['Version 1'],
  )
})
