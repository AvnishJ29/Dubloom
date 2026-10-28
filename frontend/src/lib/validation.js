/** Accept seconds, m:ss or h:mm:ss; never silently turn malformed input into zero. */
export function parseClock(value) {
  const text = String(value).trim()
  if (
    !/^\d+(?:\.\d+)?$/.test(text) &&
    !/^\d+:\d{1,2}(?:\.\d+)?$/.test(text) &&
    !/^\d+:\d{1,2}:\d{1,2}(?:\.\d+)?$/.test(text)
  )
    return NaN
  const parts = text.split(':').map(Number)
  if (parts.slice(1).some((part) => part >= 60)) return NaN
  const seconds = parts.reduce((total, part) => total * 60 + part, 0)
  return Number.isFinite(seconds) ? seconds : NaN
}
/** True for an absolute http(s) URL with a host. */
export function validVideoUrl(value) {
  try {
    const url = new URL(value.trim())
    return ['http:', 'https:'].includes(url.protocol) && !!url.hostname
  } catch {
    return false
  }
}
