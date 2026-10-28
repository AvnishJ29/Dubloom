/** A dub's history for the sidebar, newest first: one entry per change the owner made (script edits, voices, takes,
 *  merges), plus the first dub. A studio re-render changes nothing the owner made, so it does not get an entry of its
 *  own: it only replaces the render of the change before it, and that entry plays the newest render. Every archived
 *  render stays in versions/<n>/; this only decides what is listed. */
const STUDIO = new Set(['Re-rendered', 'Improved lip sync and timing'])

export function generations(job) {
  const versions = job?.versions || []
  const renders = versions.map((version, i) => ({
    number: version.number,
    reason: versions[i - 1]?.reason,
    at: version.rendered_at,
  }))
  if (job?.status === 'completed' && job.video_url)
    renders.push({ number: 0, latest: true, reason: versions.at(-1)?.reason, at: job.finished_at })
  const entries = []
  for (const render of renders) {
    const studio = entries.length && STUDIO.has(render.reason)
    if (studio)
      Object.assign(entries.at(-1), {
        number: render.number,
        latest: !!render.latest,
        at: render.at,
      }) // the newer render of the same change
    else
      entries.push({
        number: render.number,
        latest: !!render.latest,
        made: render.reason || 'First dub',
        at: render.at,
      })
  }
  entries.forEach((entry, i) => {
    entry.name = entry.latest ? 'Latest' : `Version ${i + 1}`
  })
  return entries.reverse()
}
