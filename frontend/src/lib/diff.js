/** Word-level diff of two lines: [{ op: 'same' | 'del' | 'ins', text }], each token carrying its trailing space,
 *  so the editor can show exactly which words a suggestion removes and adds. */
export function diffWords(before, after) {
  const a = (before || '').trim().split(/\s+/).filter(Boolean)
  const b = (after || '').trim().split(/\s+/).filter(Boolean)
  const n = a.length,
    m = b.length
  const lcs = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0))
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1])
  const parts = []
  let i = 0,
    j = 0
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) {
      parts.push({ op: 'same', text: a[i] + ' ' })
      i++
      j++
    } else if (j < m && (i >= n || lcs[i][j + 1] >= lcs[i + 1][j])) {
      parts.push({ op: 'ins', text: b[j] + ' ' })
      j++
    } else {
      parts.push({ op: 'del', text: a[i] + ' ' })
      i++
    }
  }
  return parts
}
