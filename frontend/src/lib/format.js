/** Pipeline stages in order, as the progress view shows them; `key` matches the job's `stage_key`. */
export const STAGES = [
  { key: 'source', label: 'Fetching the video', short: 'Video', icon: 'bi-cloud-arrow-down' },
  { key: 'audio', label: 'Separating voices from music', short: 'Audio', icon: 'bi-soundwave' },
  { key: 'speakers', label: "Finding who's speaking", short: 'Speakers', icon: 'bi-people' },
  {
    key: 'transcribe',
    label: 'Writing down the dialogue',
    short: 'Transcript',
    icon: 'bi-text-left',
  },
  { key: 'translate', label: 'Translating the script', short: 'Translation', icon: 'bi-translate' },
  { key: 'voice', label: 'Recording the new voices', short: 'Voices', icon: 'bi-mic' },
  { key: 'mix', label: 'Mixing with the original sound', short: 'Mix', icon: 'bi-sliders' },
  {
    key: 'lipsync',
    label: 'Matching lips to the new voices',
    short: 'Lip sync',
    icon: 'bi-person-video3',
  },
]

/** Plain-language names for the lip-sync engines the API exposes. */
export const LIP_MODEL_COPY = {
  latentsync: { label: 'Best quality', note: 'Natural mouths. Uses the GPU.' },
  wav2lip: { label: 'Fast', note: 'Softer mouth detail. Runs anywhere.' },
  wav2lip_gan: { label: 'Fast, sharper', note: 'Crisper teeth and lips.' },
  none: {
    label: 'Voice only (fastest)',
    note: 'Dubbed voice over the original picture, mouths unchanged. Minutes, no GPU.',
  },
}

/** Speaker colours, assigned in order of first appearance. */
export const PALETTE = [
  '#e8456b',
  '#2f7dd1',
  '#1f9d63',
  '#8a4fb5',
  '#d9890b',
  '#2a9fa6',
  '#c43a8f',
  '#5b7a2a',
  '#a4612c',
  '#2a4f9a',
]

/** What the page says while a job runs: the stage in plain words, plus why it may be paused. */
export function friendlyStage(job) {
  if (!job) return ''
  if (job.status === 'queued')
    return job.queue_position
      ? `Waiting to start · ${job.queue_position} ahead`
      : 'Waiting to start'
  const base = STAGES.find((stage) => stage.key === job.stage_key)?.label || job.stage || 'Working'
  const raw = (job.stage || '').toLowerCase()
  if (raw.includes('rate limit') || raw.includes('waiting to retry'))
    return `${base} · paused by the voice service, continues on its own`
  if (job.waiting || raw.includes('render slot')) return `${base} · waiting for the GPU`
  if (raw.startsWith('finding faces')) return 'Finding faces on screen'
  if (raw.startsWith('finding who speaks')) return 'Checking which face is speaking'
  if (raw.startsWith('preparing the preview')) return 'Preparing the preview'
  return base
}

/** Short quality name for a picture height in pixels (4K, Full HD, 480p, ...). */
export function qualityLabel(height) {
  if (!height) return ''
  if (height >= 2160) return '4K'
  if (height >= 1440) return '1440p'
  if (height >= 1080) return 'Full HD'
  if (height >= 720) return 'HD'
  return `${height}p`
}

/** Whole seconds as mm:ss, or h:mm:ss from an hour on. */
export function timecode(seconds) {
  const n = Math.max(0, Math.floor(seconds || 0))
  const h = Math.floor(n / 3600),
    m = Math.floor((n % 3600) / 60),
    s = n % 60
  return (h ? h + ':' : '') + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0')
}

/** Compact duration such as 45s, 3m 20s or 1h 5m; '' when unknown. */
export function durationLabel(seconds) {
  if (seconds == null || Number.isNaN(seconds)) return ''
  if (seconds < 60) return `${Math.round(seconds)}s`
  const m = Math.floor(seconds / 60),
    s = Math.round(seconds % 60)
  if (m < 60) return s ? `${m}m ${s}s` : `${m}m`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

/** File size in B, KB, MB or GB (binary units); '' for zero or unknown. */
export function bytes(n) {
  if (!n) return ''
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024
    i++
  }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`
}

/** Short local date (month and day); '' when missing. */
export function date(value) {
  return value
    ? new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    : ''
}

/** Short local date and time; '' when missing. */
export function dateTime(value) {
  return value
    ? new Date(value).toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
      })
    : ''
}

/** The speaker's reviewed name, else "Speaker N" from a SPEAKER_xx id (numbered from 1). */
export function speakerLabel(id, profiles) {
  const name = profiles?.[id]?.name
  if (name) return name
  const match = /SPEAKER_(\d+)/.exec(id || '')
  return match ? `Speaker ${parseInt(match[1], 10) + 1}` : id || ''
}

/** The job's speaker ids in order of first appearance. */
export function speakerOrder(job) {
  const profiles = job?.speaker_profiles || {}
  const ids = Object.keys(job?.speaker_voices || profiles)
  return ids.sort((a, b) => (profiles[a]?.first_seen ?? 0) - (profiles[b]?.first_seen ?? 0))
}

/** A PALETTE colour per speaker id, stable for a job. */
export function speakerColors(job) {
  const colors = {}
  speakerOrder(job).forEach((id, index) => {
    colors[id] = PALETTE[index % PALETTE.length]
  })
  return colors
}

/** Icon and label describing what a typed source URL points at, or null when blank or unrecognised. */
export function sourceKind(url) {
  const value = (url || '').trim().toLowerCase()
  if (!value) return null
  if (/youtube\.com|youtu\.be/.test(value)) return { icon: 'bi-youtube', label: 'YouTube video' }
  if (/vimeo\.com/.test(value)) return { icon: 'bi-vimeo', label: 'Vimeo video' }
  if (/\.(mp4|m4v|mkv|mov|webm|avi|mpg|mpeg|ts|flv|wmv|3gp|ogv)(\?|#|$)/.test(value))
    return { icon: 'bi-file-earmark-play', label: 'Video file' }
  if (/\.(m3u8|mpd)(\?|#|$)/.test(value)) return { icon: 'bi-broadcast', label: 'Live stream' }
  if (/^https?:\/\//.test(value)) return { icon: 'bi-globe2', label: 'Web page' }
  return null
}

/** Every language in its own script, next to the English name the API uses. */
export const NATIVE_NAMES = {
  as: 'অসমীয়া',
  bn: 'বাংলা',
  brx: 'बड़ो',
  doi: 'डोगरी',
  en: 'English',
  gu: 'ગુજરાતી',
  hi: 'हिन्दी',
  kn: 'ಕನ್ನಡ',
  kok: 'कोंकणी',
  ks: 'کٲشُر',
  mai: 'मैथिली',
  ml: 'മലയാളം',
  mni: 'ꯃꯤꯇꯩꯂꯣꯟ',
  mr: 'मराठी',
  ne: 'नेपाली',
  or: 'ଓଡ଼ିଆ',
  pa: 'ਪੰਜਾਬੀ',
  sa: 'संस्कृतम्',
  sat: 'ᱥᱟᱱᱛᱟᱲᱤ',
  sd: 'سنڌي',
  ta: 'தமிழ்',
  te: 'తెలుగు',
  ur: 'اردو',
}
