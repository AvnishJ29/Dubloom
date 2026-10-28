/** Live timing for a line being edited, from the pace of the voice that speaks it, with the limits the mix applies:
 *  up to 1.15x it fits (borrowing the pause after it if needed), up to 1.3x it is rushed, beyond that its end is cut. */
const PACE_MAX = 1.15,
  PACE_HARD_MAX = 1.3,
  GAP_MARGIN = 0.15

/** Characters per second for every line: its own voiced clip, blended with its voice's median for short lines. */
export function paces(segments) {
  const own = {},
    bySpeaker = {}
  segments.forEach((line, index) => {
    const slot = line.end - line.start
    if (!line.voiced_text || !line.fit_ratio || slot <= 0) return
    own[index] = line.voiced_text.length / (line.fit_ratio * slot)
    ;(bySpeaker[line.speaker] ||= []).push(own[index])
  })
  const median = (list) => [...list].sort((a, b) => a - b)[Math.floor(list.length / 2)]
  return segments.map((line, index) => {
    const voice = bySpeaker[line.speaker]?.length ? median(bySpeaker[line.speaker]) : null
    if (own[index] == null) return voice
    const weight = Math.min(1, (line.end - line.start) / 8)
    return voice == null ? own[index] : own[index] * weight + voice * (1 - weight)
  })
}

/** For every line, when the next line starts (Infinity for the last): computed once per script, not per keystroke. */
export function nextStarts(segments) {
  const starts = segments.map((line) => line.start).sort((a, b) => a - b)
  return segments.map((line) => starts.find((start) => start > line.start + 0.01) ?? Infinity)
}

/** { kind: 'ok' | 'warn' | 'bad', text } for `text` spoken in line `index`, or null when the pace is unknown. */
export function fitLine(segments, rates, index, text, nexts = nextStarts(segments)) {
  const line = segments[index],
    rate = rates[index],
    words = (text || '').trim()
  if (!line || !rate || !words) return null
  const slot = line.end - line.start
  const next = nexts[index]
  const room = Math.max(slot, (Number.isFinite(next) ? next - line.start : slot + 2) - GAP_MARGIN)
  const seconds = words.length / rate
  const over = Math.ceil(words.length - room * PACE_MAX * rate)
  const head = `≈ ${seconds.toFixed(1)} s for a ${slot.toFixed(1)} s slot`
  if (seconds <= slot * PACE_MAX + 0.1) return { kind: 'ok', text: `${head} · fits` }
  if (seconds <= room * PACE_MAX)
    return { kind: 'ok', text: `${head} · fits, using the pause after it` }
  if (seconds <= room * PACE_HARD_MAX)
    return {
      kind: 'warn',
      text: `${head} · spoken faster (${(seconds / room).toFixed(2)}x); about ${over} fewer characters would sound natural`,
    }
  return {
    kind: 'bad',
    text: `${head} · too long: the last words will be cut. Remove about ${over} characters`,
  }
}

/** A placed line worth the user's attention: its last words were cut, or it is spoken clearly faster than the
 *  speaker. Renders before 26 Oct have no `timing_trimmed`; a speed at the 1.3x ceiling means the same. */
export function timingIssue(line) {
  const speed = line?.timing_ratio || 1
  if (line?.timing_trimmed ?? speed >= PACE_HARD_MAX - 0.001)
    // the flag when the render recorded it; older renders by speed
    return {
      kind: 'bad',
      short: 'last words cut',
      text: `Last words cut: this line is ${(line.fit_ratio || speed).toFixed(1)}x as long as the original. Shorten it.`,
    }
  if (speed > PACE_MAX + 0.01)
    return {
      kind: 'warn',
      short: `spoken fast · ${speed.toFixed(2)}x`,
      text: `Spoken at ${speed.toFixed(2)}x to fit. A shorter line would sound more natural.`,
    }
  return null
}
