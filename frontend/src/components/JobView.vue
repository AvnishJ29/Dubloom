<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { post, request } from '../lib/api'
import {
  LIP_MODEL_COPY,
  STAGES,
  dateTime,
  durationLabel,
  friendlyStage,
  qualityLabel,
  speakerColors,
  speakerLabel,
  timecode,
} from '../lib/format'
import { timingIssue } from '../lib/timing'
import { generations } from '../lib/history'
import StageTimeline from './StageTimeline.vue'
import SpeakerPanel from './SpeakerPanel.vue'
import Transcript from './Transcript.vue'
import Shorts from './Shorts.vue'

const props = defineProps({ job: Object, config: Object, version: { type: Number, default: 0 } }) // version 0 = the latest render
const emit = defineEmits(['refresh', 'reuse', 'notify', 'deleted', 'version'])
const busy = ref(false)
const rewriting = ref(false)
const taking = ref(false)
const cutting = ref(false)
const takes = ref({})
watch(
  () => props.job.takes,
  (value) => {
    takes.value = { ...(value || {}) }
  },
  { immediate: true, deep: true },
)
const proposals = ref(null)
const mode = ref('dubbed')
const tab = ref('script')
const renaming = ref(false)
const newTitle = ref('')
function startRename() {
  newTitle.value = title.value
  renaming.value = true
}
async function saveRename() {
  const value = newTitle.value.trim()
  renaming.value = false
  if (!value || value === title.value) return
  try {
    await request(`/jobs/${props.job.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: value }),
    })
    emit('refresh')
  } catch (error) {
    emit('notify', error.message || 'The name could not be changed')
  }
}
const timingIssues = computed(() =>
  (props.job.segments || [])
    .map((line) => ({ line, issue: timingIssue(line) }))
    .filter((item) => item.issue),
)
const tabs = computed(() =>
  [
    {
      id: 'script',
      label: 'Script',
      count: props.job.segments?.length || 0,
      show: !!props.job.segments?.length,
    },
    {
      id: 'speakers',
      label: 'Speakers',
      count: props.job.speaker_count || Object.keys(props.job.speaker_voices || {}).length,
      show: !!props.job.speaker_voices,
    },
    { id: 'shorts', label: 'Shorts', count: props.job.shorts?.length || 0, show: completed.value },
  ].filter((item) => item.show),
)
const player = ref(null)
const currentTime = ref(0)
const pendingTime = ref(null)
const now = ref(Date.now())
let wasPlaying = false
let ticker = null

const active = computed(() => ['queued', 'running'].includes(props.job.status))
const completed = computed(() => props.job.status === 'completed')
const stopped = computed(() => ['failed', 'cancelled'].includes(props.job.status))
const colors = computed(() => speakerColors(props.job))
const strip = ref(null)
const pct = (seconds, width = false) =>
  `${Math.max(width ? 0.4 : 0, (100 * (seconds || 0)) / (props.job.duration || 1))}%`
const syncedStarts = computed(
  () =>
    new Set(
      (props.job.lip_sync?.lines || [])
        .filter((line) => line.lip_synced)
        .map((line) => Math.round(line.start * 10)),
    ),
)
const syncedLine = (row) => syncedStarts.value.has(Math.round(row.start * 10))
const lineVerdicts = computed(() => {
  // why a line was left as filmed, by its start time
  const out = new Map()
  for (const line of props.job.lip_sync?.lines || []) {
    let why = 'lip-synced'
    if (!line.lip_synced) {
      const faces = line.faces || []
      if (!faces.length) why = 'no face on screen'
      else if (faces.some((f) => f.note && f.note.includes('profile'))) why = 'face in profile'
      else why = 'face not seen speaking'
    }
    out.set(Math.round(line.start * 10), why)
  }
  return out
})
const whyLine = (row) =>
  lineVerdicts.value.get(Math.round(row.start * 10)) ||
  (props.job.lip_sync?.engine === 'none' ? 'voice only' : 'not rendered yet')
const stripSummary = computed(() => {
  const total = props.job.segments?.length || 0,
    synced = (props.job.segments || []).filter(syncedLine).length
  if (props.job.lip_sync?.engine === 'none')
    return `${total} lines, voice only: the mouths are the original picture.`
  if (!props.job.lip_sync?.lines)
    return `${total} lines. Bright bars will mark the lines whose lips are synced.`
  return `${synced} of ${total} lines lip-synced (bright). ${total - synced ? `${total - synced} left as filmed: off screen or no clear speaking face.` : ''} Click a bar to jump.`
})
function scrub(event) {
  // click anywhere on the strip to jump there
  const rect = strip.value?.getBoundingClientRect()
  if (!rect || !props.job.duration) return
  seek(Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * props.job.duration)
}
const voices = computed(() => props.config?.voices?.[props.job.language] || [])
const languageName = (code) => (code && props.config?.languages?.[code]) || code
const title = computed(() => props.job.title || `${languageName(props.job.language)} dub`)
const hasDubbed = computed(() => completed.value && !!props.job.video_url)
const hasReview = computed(() => !hasDubbed.value && !!props.job.review_url) // the dub so far, while the lips render
const hasSource = computed(() => !!props.job.source_video_url)
const showingSource = computed(
  () =>
    mode.value === 'original' || (mode.value === 'dubbed' && !hasDubbed.value && !hasReview.value),
) // nothing dubbed to show yet
const stamp = computed(() =>
  encodeURIComponent(props.job.finished_at || props.job.updated_at || ''),
) // busts the browser cache after a re-render
const videoSrc = computed(() => {
  if (mode.value === 'dubbed' && props.version)
    return `/api/jobs/${props.job.id}/versions/${props.version}/preview`
  if (mode.value === 'dubbed' && hasDubbed.value)
    return `/api/jobs/${props.job.id}/preview?v=${stamp.value}`
  if (mode.value === 'dubbed' && hasReview.value)
    return `${props.job.review_url}?v=${encodeURIComponent(props.job.review_version || '')}`
  return hasSource.value ? props.job.source_video_url : ''
}) // streams the 1080p rendition; Download keeps the full-size master
const stageLabel = computed(
  () => STAGES.find((stage) => stage.key === props.job.stage_key)?.label || props.job.stage,
)
const stageText = computed(() => friendlyStage(props.job))
const elapsed = computed(() => {
  if (!props.job.started_at) return null
  const end =
    completed.value || stopped.value
      ? new Date(props.job.finished_at || props.job.updated_at).getTime()
      : now.value
  return Math.max(0, (end - new Date(props.job.started_at).getTime()) / 1000)
})
const totalStageSeconds = computed(() =>
  Object.values(props.job.stage_seconds || {}).reduce((sum, value) => sum + value, 0),
)
const voiceNames = computed(() => {
  // the voices the dub was generated with, in speaking order
  const names = Object.values(props.job.speaker_voices || {})
  return names.length
    ? 'voices: ' +
        (names.length > 3
          ? names.slice(0, 3).join(', ') + ` +${names.length - 3}`
          : names.join(', '))
    : ''
})
const reuseNote = computed(() => {
  // what a re-render could skip: cached lip-sync spans and their restoration
  const restore = props.job.lip_sync?.mouth_enhanced
  const reused = props.job.lip_sync?.reused_frames,
    synced = props.job.lip_sync?.synced_frames
  if (restore?.spans) return `${restore.spans_reused} of ${restore.spans} lip-sync spans reused`
  if (reused && synced) return `${Math.round((100 * reused) / synced)}% of the lips reused`
  return ''
})
const lipModelLabel = computed(
  () => LIP_MODEL_COPY[props.job.lip_sync_model]?.label || props.job.lip_sync?.model || '',
)
const warnings = computed(() => [
  ...(props.job.review_warnings || []).map((text) => ({ icon: 'bi-people', text })),
  ...(props.job.lip_sync?.warnings || []).map((text) => ({ icon: 'bi-person-video3', text })),
])
const activeIndex = computed(() => {
  const segments = props.job.segments || []
  const t = currentTime.value
  let found = -1
  for (let i = 0; i < segments.length; i++)
    if (segments[i].start <= t && t < segments[i].end) return i
  for (let i = 0; i < segments.length; i++) if (segments[i].start <= t) found = i
  return found
})
const stats = computed(() => {
  if (!completed.value) return []
  const lines = props.job.segments?.length || 0
  const synced = props.job.lip_sync?.lines_lip_synced
  const offScreen = (props.job.lip_sync?.off_screen_speakers || []).length
  return [
    {
      label: 'Speakers',
      value: props.job.speaker_count ?? 0,
      note: voiceNames.value || `${lines} line${lines === 1 ? '' : 's'}`,
    },
    ...(synced != null
      ? [
          {
            label: 'Lips synced',
            value: `${synced} of ${lines}`,
            note:
              synced < lines
                ? offScreen
                  ? 'the rest are spoken off screen'
                  : 'the rest have no clear speaking face'
                : 'every line',
          },
        ]
      : []),
    ...(props.job.fit_summary
      ? [
          {
            label: 'Lines on time',
            value: `${props.job.fit_summary.in_band} of ${props.job.fit_summary.total}`,
            note: 'spoken at a natural pace',
          },
        ]
      : []),
    { label: 'Length', value: timecode(props.job.duration), note: qualityLabel(props.job.height) },
    {
      label: 'Took',
      value: durationLabel(totalStageSeconds.value || elapsed.value),
      note: reuseNote.value,
    },
  ]
})

watch(
  () => props.version,
  () => {
    const el = player.value
    pendingTime.value = el ? el.currentTime : 0
    wasPlaying = !!el && !el.paused
    mode.value = 'dubbed'
  },
)
const earlier = computed(() =>
  props.version ? generations(props.job).find((row) => row.number === props.version) : null,
)
watch(
  () => props.job.id,
  () => {
    mode.value = 'dubbed'
    currentTime.value = 0
    pendingTime.value = null
    proposals.value = null
    tab.value = 'script'
  },
)
watch(
  () => props.job.review_version,
  (next, prev) => {
    if (next && prev && mode.value === 'dubbed') {
      const el = player.value
      pendingTime.value = el ? el.currentTime : currentTime.value
      wasPlaying = !!el && !el.paused
    }
  },
)
watch(
  active,
  (value) => {
    clearInterval(ticker)
    if (value)
      ticker = setInterval(() => {
        now.value = Date.now()
      }, 1000)
  },
  { immediate: true },
)
onBeforeUnmount(() => clearInterval(ticker))

function jumpToLine(line) {
  tab.value = 'script'
  seek(line.dub_start ?? line.start)
}
function switchMode(next) {
  if (next === mode.value) return
  const element = player.value
  pendingTime.value = element ? element.currentTime : 0
  wasPlaying = !!element && !element.paused
  mode.value = next
}
function restore(event) {
  if (pendingTime.value == null) return
  event.target.currentTime = pendingTime.value
  pendingTime.value = null
  if (wasPlaying) event.target.play().catch(() => {})
}
function seek(seconds) {
  const element = player.value
  if (!element) return
  element.currentTime = seconds
  element.play().catch(() => {})
  element.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
}
function pauseOthers(event) {
  document.querySelectorAll('video, audio').forEach((media) => {
    if (media !== event.target) media.pause()
  })
}
async function act(label, work) {
  if (busy.value) return
  busy.value = true
  try {
    await work()
    emit('refresh')
  } catch (error) {
    emit('notify', error.message || `${label} failed`)
  } finally {
    busy.value = false
  }
}
const cancel = () => act('Stop', () => post(`/jobs/${props.job.id}/cancel`))
const resume = () => act('Continue', () => post(`/jobs/${props.job.id}/resume`))
const revoice = (changes) =>
  act('Voice change', () => post(`/jobs/${props.job.id}/revoice`, changes))
const merge = ({ speaker, into }) => {
  if (
    window.confirm(
      'Merge these two speakers into one? The moved lines are voiced again with that voice.',
    )
  )
    act('Merge', () => post(`/jobs/${props.job.id}/speakers/${speaker}/merge`, { into }))
}
const styleLabel = (id) => (props.config?.styles || []).find((s) => s.id === id)?.label || id
const applyScript = (payload) => {
  proposals.value = null
  return act('Script change', () => post(`/jobs/${props.job.id}/script`, payload))
}
async function addTake(request) {
  if (taking.value) return
  taking.value = true
  try {
    const result = await post(`/jobs/${props.job.id}/lines/${request.index}/takes`, {
      style: request.style,
      text: request.text,
    })
    takes.value = { ...takes.value, [request.index]: result.takes }
  } catch (error) {
    emit('notify', error.message || 'The take could not be voiced')
  } finally {
    taking.value = false
  }
}
async function pickTake(request) {
  await act('Choose take', async () => {
    await post(`/jobs/${props.job.id}/lines/${request.index}/take`, { number: request.number })
  })
}
const rerender = () => act('Re-render', () => post(`/jobs/${props.job.id}/revoice`, {}))
async function createShort(request) {
  if (cutting.value) return
  cutting.value = true
  try {
    await post(`/jobs/${props.job.id}/shorts`, request)
    emit('refresh')
    emit('notify', 'Short created.', 'success')
  } catch (error) {
    emit('notify', error.message || 'The short could not be cut')
  } finally {
    cutting.value = false
  }
}
const removeShort = (number) =>
  act('Delete short', () => request(`/jobs/${props.job.id}/shorts/${number}`, { method: 'DELETE' }))
async function proposeRewrite(request) {
  if (rewriting.value) return
  rewriting.value = true
  try {
    proposals.value = await post(`/jobs/${props.job.id}/rewrite`, request)
    if (!proposals.value.lines?.length)
      emit('notify', `The AI found nothing to change for “${request.instruction}”.`)
  } catch (error) {
    emit('notify', error.message || 'The AI could not change the script')
  } finally {
    rewriting.value = false
  }
}
async function remove() {
  if (!window.confirm(`Delete “${title.value}” and its video?`)) return
  await act('Delete', async () => {
    await request(`/jobs/${props.job.id}`, { method: 'DELETE' })
    emit('deleted')
  })
}
const subtitle = (track, fmt) =>
  `/api/jobs/${props.job.id}/subtitles?track=${track}&fmt=${fmt}&download=true`
</script>

<template>
  <div class="job" :key="job.id">
    <header class="job-head">
      <div class="job-title">
        <span v-if="!completed" :class="['status-line', 'status-' + job.status]">
          <i class="status-dot"></i>
          {{ active ? (job.status === 'queued' ? 'Waiting to start' : 'Dubbing') : job.status === 'cancelled' ? 'Stopped' : 'Needs attention' }}
        </span>
        <h1 v-if="!renaming" class="job-name">{{ title }} <button type="button" class="icon-button tiny" title="Rename this dub" aria-label="Rename this dub" @click="startRename"><i class="bi bi-pencil"></i></button></h1>
        <form v-else class="rename" @submit.prevent="saveRename">
          <input v-model="newTitle" type="text" maxlength="140" aria-label="New name" autofocus @keydown.esc="renaming = false" @blur="saveRename" />
          <button type="submit" class="button primary small">Save</button>
        </form>
        <p class="job-meta">
          <span class="job-pair">{{ languageName(job.source_language) || 'Auto' }}<i class="bi bi-arrow-right" aria-hidden="true"></i>{{ languageName(job.language) }}</span>
          <span v-if="job.duration">{{ timecode(job.duration) }}</span>
          <span v-if="job.height">{{ qualityLabel(job.height) }}</span>
          <span v-if="job.background_mode">{{ { separate: 'Music kept', duck: 'Original sound lowered', none: 'No background' }[job.background_mode] }}</span>
          <span v-if="lipModelLabel">Lip sync: {{ lipModelLabel.toLowerCase() }}</span>
          <span v-if="job.tts_engine === 'clone'">Own voices</span>
          <span v-if="job.delivery_style && job.delivery_style !== 'neutral'">{{ styleLabel(job.delivery_style) }}</span>
        </p>
        <p class="job-sub">
          <span v-if="job.script_profile?.label" :title="job.script_profile.register">{{ job.script_profile.label }}</span>
          <span>{{ job.finished_at ? 'Dubbed' : 'Created' }} {{ dateTime(job.finished_at || job.updated_at || job.created_at) }}</span>
          <a v-if="job.source_url" :href="job.source_url" target="_blank" rel="noopener">Original<i class="bi bi-arrow-up-right" aria-hidden="true"></i></a>
          <span v-else-if="job.source_name">{{ job.source_name }}</span>
        </p>
      </div>
      <div class="job-actions">
        <template v-if="completed">
          <a class="button primary" :href="job.video_url + '?download=true'" :download="'dubloom-' + job.id + '.mp4'"><i class="bi bi-download"></i> Download video</a>
          <details class="menu">
            <summary class="button ghost"><i class="bi bi-badge-cc"></i> Subtitles <i class="bi bi-chevron-down"></i></summary>
            <div class="menu-list">
              <a :href="subtitle('target', 'srt')" download>{{ languageName(job.language) }} · SRT</a>
              <a :href="subtitle('target', 'vtt')" download>{{ languageName(job.language) }} · WebVTT</a>
              <a :href="subtitle('source', 'srt')" download>Original · SRT</a>
              <a :href="subtitle('source', 'vtt')" download>Original · WebVTT</a>
            </div>
          </details>
        </template>
        <button v-if="active" class="button danger-ghost" :disabled="busy" @click="cancel"><i class="bi bi-stop-circle"></i> Stop</button>
        <button v-if="stopped" class="button primary" :disabled="busy" @click="resume"><i class="bi bi-play-circle"></i> Continue</button>
        <button class="button ghost" @click="emit('reuse')"><i class="bi bi-copy"></i> Dub again</button>
        <button v-if="!active" class="icon-button danger" :disabled="busy" title="Delete" aria-label="Delete this dub" @click="remove"><i class="bi bi-trash3"></i></button>
      </div>
    </header>

    <section v-if="active" class="panel progress-panel" role="status" aria-live="polite">
      <div class="progress-top">
        <div class="progress-copy">
          <h2><span class="pulse" aria-hidden="true"></span>{{ stageText }}</h2>
          <p>
            <template v-if="job.status === 'queued'">It starts as soon as a slot frees up.</template>
            <template v-else-if="job.retry_at">The voice service asked for a short pause. It picks up again at {{ new Date(job.retry_at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) }}.</template>
            <template v-else>You can leave this page; it keeps going.</template>
            <span v-if="elapsed != null">{{ ' ' + durationLabel(elapsed) }} so far.</span>
          </p>
          <p v-if="job.background?.mode === 'separate' && ['queued', 'running'].includes(job.background.status)" class="progress-side"><i class="bi bi-music-note-list"></i> Lifting the voices off the music{{ job.background.status === 'running' ? ` · ${Math.round((job.background.progress || 0) * 100)}%` : '' }}</p>
        </div>
        <strong class="progress-value">{{ Math.round(job.progress || 0) }}<small>%</small></strong>
      </div>
      <div class="bar" role="progressbar" :aria-valuenow="Math.round(job.progress || 0)" aria-valuemin="0" aria-valuemax="100"><div :style="{ width: (job.progress || 0) + '%' }"></div></div>
      <StageTimeline :stage-key="job.stage_key" :status="job.status" :stage-seconds="job.stage_seconds || {}" />
    </section>

    <section v-else-if="stopped" class="panel stopped-panel">
      <div class="stopped-copy">
        <i :class="'bi ' + (job.status === 'cancelled' ? 'bi-stop-circle' : 'bi-exclamation-octagon')"></i>
        <div>
          <h2>{{ job.status === 'cancelled' ? 'Stopped' : 'Something went wrong' }}<template v-if="job.stage_key"> while {{ stageLabel.toLowerCase() }}</template></h2>
          <p>{{ job.error || 'Everything finished so far is saved. Continue to pick up where it stopped.' }}</p>
          <p v-if="job.error" class="stopped-hint">Press Continue to pick up where it stopped. Finished steps are kept.</p>
        </div>
      </div>
      <StageTimeline :stage-key="job.stage_key" :status="job.status" :stage-seconds="job.stage_seconds || {}" />
    </section>

    <div v-if="hasSource || hasDubbed || hasReview" class="media-grid">
      <section class="stage">
        <header class="stage-head">
          <div v-if="hasDubbed || hasReview" class="segmented" role="tablist">
            <button role="tab" :aria-selected="mode === 'dubbed' && !version" :class="{ active: mode === 'dubbed' && !version }" @click="version ? emit('version', 0) : switchMode('dubbed')">{{ version ? 'Latest' : 'Dubbed' }}</button>
            <button v-if="version" role="tab" :aria-selected="mode === 'dubbed'" :class="{ active: mode === 'dubbed' }" @click="switchMode('dubbed')">{{ earlier?.name || 'Earlier version' }}</button>
            <button role="tab" :aria-selected="showingSource" :class="{ active: showingSource }" @click="switchMode('original')">Original</button>
          </div>
          <h2 v-else>Original video</h2>
          <span v-if="mode === 'dubbed' && earlier" class="stage-sub">{{ earlier.name }} · {{ earlier.made }} · {{ dateTime(earlier.at) }} · <a :href="`/api/jobs/${job.id}/versions/${version}/video?download=true`" download>Download</a></span>
          <span v-else class="stage-sub">{{ mode === 'dubbed' && hasDubbed ? 'Preview. The download is full quality.' : mode === 'dubbed' && hasReview ? `The dub so far: ${job.review_note || 'dubbed voice over the original picture'}. Updates as spans finish.` : 'As it came in' }}</span>
        </header>
        <video ref="player" :key="videoSrc" :src="videoSrc" :poster="job.poster_url || undefined" controls playsinline preload="metadata"
               aria-label="Video" @timeupdate="currentTime = $event.target.currentTime" @loadedmetadata="restore" @play="pauseOthers"></video>
        <div v-if="job.duration && job.segments?.length" ref="strip" class="timeline-strip" role="group" aria-label="Lines on the timeline: click to jump" @click="scrub">
          <span v-for="(row, i) in job.segments" :key="i" class="tl-line" :class="{ synced: syncedLine(row), active: i === activeIndex }"
                :style="{ left: pct(row.start), width: pct((row.dub_end || row.end) - row.start, true), '--speaker': colors[row.speaker] }"
                :title="`${speakerLabel(row.speaker, job.speaker_profiles)} · ${timecode(row.start)} · ${whyLine(row)}`"></span>
          <span class="tl-head" :style="{ left: pct(currentTime) }"></span>
        </div>
        <small v-if="job.duration && job.segments?.length" class="timeline-key">{{ stripSummary }}</small>
        <div v-if="completed && job.audio_url" class="audio-row">
          <i class="bi bi-soundwave"></i>
          <div class="audio-copy"><strong>Dubbed audio only</strong><small>The new voices mixed with the original music</small></div>
          <audio :src="job.audio_url + '?v=' + stamp" controls preload="none" aria-label="Dubbed audio" @play="pauseOthers"></audio>
          <a class="icon-button" :href="job.audio_url + '?download=true'" download title="Download audio" aria-label="Download audio"><i class="bi bi-download"></i></a>
        </div>
      </section>
      <aside class="side-column">
        <section v-if="stats.length" class="panel stats">
          <div v-for="item in stats" :key="item.label" class="stat">
            <span class="stat-label">{{ item.label }}</span>
            <strong class="stat-value">{{ item.value }}</strong>
            <small v-if="item.note" class="stat-note">{{ item.note }}</small>
          </div>
        </section>
        <section v-if="warnings.length || timingIssues.length" class="panel notes">
          <h3><i class="bi bi-journal-check"></i> Worth a look</h3>
          <ul><li v-for="(item, index) in warnings" :key="index"><i :class="'bi ' + item.icon"></i><span>{{ item.text }}</span></li></ul>
          <template v-if="timingIssues.length">
            <p class="notes-sub" title="Edit a line with its pencil: the meter shows when it fits."><i class="bi bi-speedometer2"></i> {{ timingIssues.length }} line{{ timingIssues.length === 1 ? '' : 's' }} to shorten</p>
            <ul class="issue-list">
              <li v-for="item in timingIssues" :key="item.line.start"><button type="button" :class="['issue', item.issue.kind]" :title="item.issue.text" @click="jumpToLine(item.line)"><span class="issue-time">{{ timecode(item.line.start) }}</span><span>{{ item.issue.short }}</span><i class="bi bi-arrow-right-short"></i></button></li>
            </ul>
          </template>
        </section>
      </aside>
    </div>

    <div v-if="completed && job.pending_takes" class="pending-bar" role="status">
      <span><i class="bi bi-mic"></i> {{ job.pending_takes }} line{{ job.pending_takes === 1 ? '' : 's' }} directed. Re-render to hear {{ job.pending_takes === 1 ? 'it' : 'them' }} in the dub; the current render is kept in History.</span>
      <button class="button primary small" :disabled="busy" @click="rerender"><i class="bi bi-arrow-repeat"></i> Re-render</button>
    </div>
    <nav v-if="tabs.length > 1" class="tabs" role="tablist" aria-label="Project sections">
      <button v-for="item in tabs" :key="item.id" role="tab" :aria-selected="tab === item.id" :class="{ active: tab === item.id }" @click="tab = item.id">{{ item.label }}<span v-if="item.count" class="count">{{ item.count }}</span></button>
    </nav>
    <Shorts v-if="completed && tab === 'shorts'" :job="job" :player-time="currentTime" :busy="cutting || busy" @create="createShort" @remove="removeShort" />
    <SpeakerPanel v-if="job.speaker_voices && (tab === 'speakers' || tabs.length <= 1)" :job="job" :voices="voices" :colors="colors" :styles="config?.styles || []" :editable="!active && !!job.source_video_url" :busy="busy" @revoice="revoice" @seek="seek" @merge="merge" />
    <Transcript v-if="job.segments?.length && (tab === 'script' || tabs.length <= 1)" :segments="job.segments" :colors="colors" :profiles="job.speaker_profiles || {}" :voices="job.speaker_voices || {}" :reviewed="!!job.script_profile" :active-index="activeIndex" :can-seek="hasSource || hasDubbed"
                :editable="!active && !!job.source_video_url && job.segments.some(s => s.translation)" :busy="busy"
                :language-name="languageName(job.language)" :proposals="proposals" :rewriting="rewriting"
                :job-id="job.id" :styles="config?.styles || []" :takes="takes" :taking="taking"
                @seek="seek" @apply="applyScript" @rewrite="proposeRewrite" @take="addTake" @choose="pickTake" />
  </div>
</template>
