<script setup>
import { parseClock, validVideoUrl } from '../lib/validation'
import LanguagePicker from './LanguagePicker.vue'
import OptionPicker from './OptionPicker.vue'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { post, upload } from '../lib/api'
import { bytes, date, friendlyStage, sourceKind, timecode } from '../lib/format'

const props = defineProps({
  config: Object,
  health: Object,
  ready: Boolean,
  prefill: Object,
  recent: { type: Array, default: () => [] },
})
const emit = defineEmits(['created', 'error', 'open'])
const copyrightYear = new Date().getFullYear()
const mode = ref('url')
const url = ref('')
const file = ref(null)
const preview = ref('')
const dragging = ref(false)
const language = ref('hi')
const sourceLanguage = ref('')
const voice = ref('')
const numSpeakers = ref('')
const deliveryStyle = ref('neutral')
const voiceMode = ref('auto')
const title = ref('')
const backgroundMode = ref('separate')
const casting = ref('auto') // auto: match voices to the speakers; manual: your voices in speaking order
const chosenVoices = ref([])
const lipModel = ref('')
const lipModels = computed(() => props.config?.lip_sync_models || [])
const heroLangs = [
  'हिन्दी',
  'தமிழ்',
  'తెలుగు',
  'বাংলা',
  'मराठी',
  'ಕನ್ನಡ',
  'മലയാളം',
  'ગુજરાતી',
  'ਪੰਜਾਬੀ',
  'ଓଡ଼ିଆ',
  'অসমীয়া',
  'नेपाली',
  'اردو',
]
const cycle = computed(() => [
  ...heroLangs.slice(0, 7),
  `${languages.value.length || 23} Indian languages`,
])
const word = ref(7) // the headline cycles through the languages, starting and ending on the count (index 7)
const prev = ref(-1)
let wordTimer = null
onMounted(() => {
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
  wordTimer = setInterval(() => {
    prev.value = word.value
    word.value = (word.value + 1) % cycle.value.length
  }, 2200)
})
onBeforeUnmount(() => {
  clearInterval(wordTimer)
  if (preview.value) URL.revokeObjectURL(preview.value)
})
// The dub card beside the headline speaks the same language as the headline: one line, voiced in each language in turn.
const SAMPLES = {
  'हिन्दी': 'सभी का फिर से स्वागत है।',
  'தமிழ்': 'அனைவரையும் மீண்டும் வரவேற்கிறோம்.',
  'తెలుగు': 'అందరికీ మళ్ళీ స్వాగతం.',
  'বাংলা': 'সবাইকে আবার স্বাগতম।',
  'मराठी': 'सर्वांचे पुन्हा स्वागत आहे.',
  'ಕನ್ನಡ': 'ಎಲ್ಲರಿಗೂ ಮತ್ತೆ ಸ್ವಾಗತ.',
  'മലയാളം': 'എല്ലാവർക്കും വീണ്ടും സ്വാഗതം.',
}
// any language into any other: the source changes with the target (English or another Indian language)
const FROM = {
  'हिन्दी': 'English',
  'தமிழ்': 'हिन्दी',
  'తెలుగు': 'বাংলা',
  'বাংলা': 'English',
  'मराठी': 'ಕನ್ನಡ',
  'ಕನ್ನಡ': 'मराठी',
  'മലയാളം': 'English',
}
const sample = computed(() => {
  const name = heroLangs[word.value % 7]
  return { name, from: FROM[name], text: SAMPLES[name] }
})
const lipLabel = (model) =>
  model.id === 'latentsync'
    ? 'Best quality (LatentSync) · recommended'
    : model.id === 'wav2lip_gan'
      ? 'Fast, sharper (Wav2Lip GAN)'
      : model.id === 'wav2lip'
        ? 'Fast (Wav2Lip)'
        : model.id === 'none'
          ? 'Voice only · fastest (no lip sync)'
          : model.label
const bestModel = computed(
  () =>
    (
      lipModels.value.find((m) => m.id === 'latentsync' && m.available) ||
      lipModels.value.find((m) => m.available && m.id !== 'none')
    )?.id || 'latentsync',
)
const fast = computed({
  get: () => lipModel.value === 'none',
  set: (value) => {
    lipModel.value = value ? 'none' : bestModel.value
  },
}) // Voice only vs lip-synced
const slots = computed(() => Math.min(6, Math.max(1, Number(numSpeakers.value) || 2)))
const partOnly = ref(false)
const clipStart = ref('0:00')
const clipEnd = ref('1:00')
const partValid = computed(
  () =>
    !partOnly.value ||
    (Number.isFinite(parseClock(clipStart.value)) &&
      Number.isFinite(parseClock(clipEnd.value)) &&
      parseClock(clipEnd.value) - parseClock(clipStart.value) >= 1 &&
      parseClock(clipEnd.value) - parseClock(clipStart.value) <=
        (props.config?.max_video_minutes || 60) * 60),
)
const modelReady = computed(() =>
  lipModels.value.some((model) => model.id === lipModel.value && model.available),
)
const lipSyncReady = computed(() =>
  lipModels.value.some((model) => model.id !== 'none' && model.available),
)
const submitting = ref(false)
const uploadProgress = ref(0)
const picker = ref(null)
const settingsOpen = ref(false)

const languages = computed(() => Object.entries(props.config?.languages || {}))
const voices = computed(() => props.config?.voices?.[language.value] || [])
const styles = computed(() => props.config?.styles || [])
const kind = computed(() => sourceKind(url.value))
const accept = computed(() => (props.config?.upload_extensions || []).join(','))
const separationReady = computed(() => !!props.config?.background?.separation_available)
const cloneReady = computed(() => (props.config?.clone_languages || []).includes(language.value))
// Picker values retain the exact types and availability rules used by the request payload.
const sourceLanguageOptions = computed(() => [
  { value: '', label: 'Detect it for me' },
  ...languages.value.map(([value, label]) => ({ value, label })),
])
const voiceModeOptions = computed(() => [
  { value: 'auto', label: 'A studio voice for each speaker' },
  { value: 'clone', label: "Each speaker's own voice", disabled: !cloneReady.value },
])
const deliveryOptions = computed(() =>
  styles.value.map((item) => ({ value: item.id, label: item.label })),
)
const castingOptions = [
  { value: 'auto', label: 'Match voices to the speakers automatically' },
  { value: 'manual', label: 'Choose the voices myself' },
]
const voiceOptions = computed(() => [
  { value: '', label: 'Match automatically' },
  ...voices.value.map((item) => ({
    value: item.name,
    label: [item.name, item.gender, item.language, item.best_for?.[0]].filter(Boolean).join(' · '),
  })),
])
const lipOptions = computed(() =>
  lipModels.value.map((model) => ({
    value: model.id,
    label: lipLabel(model) + (model.available ? '' : ' · unavailable'),
    disabled: !model.available,
  })),
)
const speakerCountOptions = [
  { value: '', label: 'Detect it for me' },
  ...Array.from({ length: 12 }, (_, index) => ({ value: index + 1, label: String(index + 1) })),
]
const backgroundOptions = computed(() => [
  {
    value: 'separate',
    label: 'Keep the music, replace the voices',
    disabled: !separationReady.value,
  },
  { value: 'duck', label: 'Keep everything, lowered under the dub' },
  { value: 'none', label: 'Silence between lines' },
])
const sourceJob = ref(null) // 'Dub again': the earlier dub whose video is reused as it was (the same part, no new download)
const composerEl = ref(null)
const canSubmit = computed(
  () =>
    props.ready &&
    modelReady.value &&
    !submitting.value &&
    (sourceJob.value
      ? true
      : partValid.value && (mode.value === 'url' ? validVideoUrl(url.value) : !!file.value)),
)
const languageName = (code) => props.config?.languages?.[code] || code

watch(language, () => {
  chosenVoices.value = []
  voice.value = ''
  if (!cloneReady.value) voiceMode.value = 'auto'
})
watch(
  () => props.config,
  (value) => {
    if (!value) return
    const models = value.lip_sync_models || []
    if (!lipModel.value)
      lipModel.value =
        (
          models.find((m) => m.id === 'latentsync' && m.available) ||
          models.find((m) => m.available)
        )?.id || '' // the best the studio can run
    if (value.background?.default)
      backgroundMode.value = value.background.separation_available
        ? value.background.default
        : value.background.default === 'separate'
          ? 'duck'
          : value.background.default
  },
  { immediate: true },
)
watch(
  () => props.prefill,
  (value) => {
    if (!value) {
      sourceJob.value = null
      return
    }
    sourceJob.value = value.id
      ? {
          id: value.id,
          title: value.title_shown || value.source_name || 'This video',
          poster: value.poster_url,
          duration: value.duration,
          part: value.clip_end ? [value.clip_start, value.clip_end] : null,
        }
      : null
    if (sourceJob.value)
      nextTick(() => composerEl.value?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
    if (value.source_url) {
      mode.value = 'url'
      url.value = value.source_url
    }
    language.value = value.language || 'hi'
    sourceLanguage.value = value.source_language || ''
    voice.value = value.voice || ''
    numSpeakers.value = value.num_speakers || ''
    if (value.background_mode) backgroundMode.value = value.background_mode
    if (value.delivery_style) deliveryStyle.value = value.delivery_style
    if (value.clip_end) {
      partOnly.value = true
      clipStart.value = `${Math.floor(value.clip_start / 60)}:${String(Math.floor(value.clip_start % 60)).padStart(2, '0')}`
      clipEnd.value = `${Math.floor(value.clip_end / 60)}:${String(Math.floor(value.clip_end % 60)).padStart(2, '0')}`
    }
    if (value.tts_engine) voiceMode.value = value.tts_engine
    title.value = value.title || ''
    settingsOpen.value = true
  },
  { immediate: true },
)

function choose(selected) {
  if (!selected) return
  if (!selected.size) {
    emit('error', 'That file is empty. Choose a video with content.')
    return
  }
  const extension = '.' + selected.name.split('.').pop().toLowerCase()
  if (props.config?.upload_extensions && !props.config.upload_extensions.includes(extension)) {
    emit('error', `“${selected.name}” is not a video type we can read.`)
    return
  }
  if (props.config && selected.size > props.config.max_upload_mb * 1024 * 1024) {
    emit('error', `That file is larger than the ${props.config.max_upload_mb} MB limit.`)
    return
  }
  if (preview.value) URL.revokeObjectURL(preview.value)
  file.value = selected
  preview.value = URL.createObjectURL(selected)
  if (!title.value) title.value = selected.name.replace(/\.[^.]+$/, '')
}
function onDrop(event) {
  dragging.value = false
  const dropped = event.dataTransfer?.files?.[0]
  if (dropped) {
    mode.value = 'file'
    choose(dropped)
  }
}
function clearFile() {
  if (preview.value) URL.revokeObjectURL(preview.value)
  file.value = null
  preview.value = ''
}
async function submit() {
  if (!canSubmit.value) return
  submitting.value = true
  uploadProgress.value = 0
  try {
    const options = {
      language: language.value,
      source_language: sourceLanguage.value || null,
      voice: voice.value || null,
      num_speakers: numSpeakers.value ? Number(numSpeakers.value) : null,
      delivery_style: deliveryStyle.value,
      background_mode: backgroundMode.value,
      tts_engine: voiceMode.value,
      title: title.value.trim() || null,
      lip_sync_model: lipModel.value || null,
      voices:
        casting.value === 'manual' ? chosenVoices.value.slice(0, slots.value).filter(Boolean) : [],
      clip_start: partOnly.value ? parseClock(clipStart.value) : null,
      clip_end: partOnly.value ? parseClock(clipEnd.value) : null,
    }
    let result
    if (sourceJob.value) {
      result = await post('/jobs', {
        source_job: sourceJob.value.id,
        ...options,
        clip_start: null,
        clip_end: null,
      }) // the saved video is already that part
    } else if (mode.value === 'url') {
      result = await post('/jobs', { url: url.value.trim(), ...options })
    } else {
      const form = new FormData()
      form.append('file', file.value)
      for (const [key, value] of Object.entries(options))
        if (value !== null && value !== '' && !(Array.isArray(value) && !value.length))
          form.append(key, Array.isArray(value) ? value.join(',') : String(value))
      result = await upload('/jobs/upload', form, (fraction) => {
        uploadProgress.value = fraction
      })
    }
    url.value = ''
    clearFile()
    title.value = ''
    sourceJob.value = null
    emit('created', result.id)
  } catch (error) {
    emit('error', error.message)
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <div class="create">
    <section class="hero">
      <div class="hero-copy">
        <span class="hero-kicker"><i class="dot" aria-hidden="true"></i>AI dubbing studio<span class="sep" aria-hidden="true">·</span>built on India's sovereign models</span>
        <h1>Your video, <span class="serif">fluent</span> in
          <span class="rotator"><em v-for="(item, i) in cycle" :key="item" :class="{ on: i === word, was: i === prev }" :aria-hidden="i !== word">{{ item }}</em></span>
        </h1>
        <p>Take your story beyond one language. Natural voices matched to every speaker, lips that follow the new words, and the full quality of your original video.</p>
        <ul class="hero-langs" aria-label="Languages">
          <li v-for="name in heroLangs" :key="name">{{ name }}</li>
          <li v-if="(languages.length || 23) > heroLangs.length" class="more">+ {{ (languages.length || 23) - heroLangs.length }} more</li>
        </ul>
      </div>
      <div class="hero-visual" aria-hidden="true">
        <div class="dub-frame">
          <img src="/hero-booth.svg" alt="" width="800" height="500" />
          <Transition name="swap" mode="out-in"><span :key="sample.name" class="dub-pill">{{ sample.from }} <i class="bi bi-arrow-right-short"></i><b>{{ sample.name }}</b></span></Transition>
          <div class="dub-caption">
            <span class="dub-wave"><i v-for="n in 16" :key="n" :style="{ animationDelay: `${(n * 137) % 900}ms` }"></i></span>
            <Transition name="swap" mode="out-in"><p :key="sample.name">{{ sample.text }}</p></Transition>
          </div>
        </div>
        <ol class="dub-pipeline">
          <li style="--i: 0"><span class="node"></span>Transcribe</li>
          <li style="--i: 1"><span class="node"></span>Translate</li>
          <li style="--i: 2"><span class="node"></span>Voice</li>
          <li style="--i: 3"><span class="node"></span>Lip-sync</li>
        </ol>
      </div>
    </section>

    <section class="features" aria-label="Why Dubloom">
      <article><span class="feature-icon"><i class="bi bi-badge-4k" aria-hidden="true"></i></span><h3>Up to 4K</h3><p>Your original resolution, kept to the last pixel.</p></article>
      <article><span class="feature-icon"><i class="bi bi-person-bounding-box" aria-hidden="true"></i></span><h3>Lips that match</h3><p>Mouths redrawn to follow every dubbed word.</p></article>
      <article><span class="feature-icon"><i class="bi bi-soundwave" aria-hidden="true"></i></span><h3>Voices that fit</h3><p>A matched voice for every speaker, in {{ languages.length || 23 }} languages.</p></article>
      <article><span class="feature-icon"><i class="bi bi-layers" aria-hidden="true"></i></span><h3>You're in control</h3><p>Edit any line, compare versions, restore anything.</p></article>
    </section>

    <form ref="composerEl" class="composer" :aria-busy="submitting" @submit.prevent="submit">
      <header class="composer-heading"><div><span class="eyebrow">CREATE A DUB</span><h2>Start with your <span class="serif">video</span></h2></div><span class="composer-step">Source → Language → Dub</span></header>
      <div v-if="sourceJob" class="composer-row">
        <div class="source-card">
          <span class="source-thumb"><img v-if="sourceJob.poster" :src="sourceJob.poster" alt="" /><i v-else class="bi bi-film" aria-hidden="true"></i></span>
          <span class="source-copy"><small>Dub again</small><strong>{{ sourceJob.title }}</strong>
            <em>{{ sourceJob.part ? `${timecode(sourceJob.part[0])}–${timecode(sourceJob.part[1])} of the video` : timecode(sourceJob.duration) }} · the same video, no new download</em></span>
          <button type="button" class="link-button" @click="sourceJob = null">Use a different video</button>
        </div>
        <LanguagePicker v-model="language" :languages="config?.languages || {}" label="into" />
        <button class="button primary large" type="submit" :disabled="!canSubmit">
          <span v-if="submitting"><i class="bi bi-arrow-repeat spin"></i> Starting…</span>
          <span v-else>Dub it</span>
        </button>
      </div>
      <div v-if="!sourceJob" class="composer-tabs" role="group" aria-label="Video source">
        <button type="button" :aria-pressed="mode === 'url'" :class="{ active: mode === 'url' }" @click="mode = 'url'"><i class="bi bi-link-45deg"></i> Paste a link</button>
        <button type="button" :aria-pressed="mode === 'file'" :class="{ active: mode === 'file' }" @click="mode = 'file'"><i class="bi bi-upload"></i> Upload a file</button>
      </div>

      <div v-if="!sourceJob && mode === 'url'" class="composer-row">
        <label class="composer-input">
          <i :class="'bi ' + (kind?.icon || 'bi-link-45deg')" aria-hidden="true"></i>
          <input v-model="url" type="url" inputmode="url" autocomplete="off" spellcheck="false" placeholder="Paste a YouTube, Vimeo or video link" aria-label="Video link" required />
        </label>
        <LanguagePicker v-model="language" :languages="config?.languages || {}" label="into" />
        <button class="button primary large" type="submit" :disabled="!canSubmit">
          <span v-if="submitting"><i class="bi bi-arrow-repeat spin"></i> Starting…</span>
          <span v-else>Dub it</span>
        </button>
      </div>

      <div v-else-if="!sourceJob" class="composer-file">
        <div v-if="!file" :class="['dropzone', { dragging }]" tabindex="0" role="button" aria-label="Choose a video file"
             @click="picker.click()" @keydown.enter.prevent="picker.click()" @keydown.space.prevent="picker.click()" @dragover.prevent="dragging = true" @dragleave="dragging = false" @drop.prevent="onDrop">
          <i class="bi bi-film" aria-hidden="true"></i>
          <strong>Drop a video here, or <u>choose a file</u></strong>
          <span>MP4, MOV, MKV, WebM · up to {{ config?.max_upload_mb }} MB and {{ config?.max_video_minutes }} minutes</span>
        </div>
        <div v-else class="file-chip">
          <video v-if="preview" :src="preview" muted playsinline preload="metadata"></video>
          <div class="file-copy"><strong>{{ file.name }}</strong><span>{{ bytes(file.size) }}</span></div>
          <button type="button" class="icon-button" aria-label="Remove file" @click="clearFile"><i class="bi bi-x-lg"></i></button>
        </div>
        <input ref="picker" type="file" :accept="accept" hidden @change="choose($event.target.files[0]); $event.target.value = ''" />
        <div class="composer-row">
          <LanguagePicker v-model="language" :languages="config?.languages || {}" label="Dub into" class="grow" />
          <button class="button primary large" type="submit" :disabled="!canSubmit">
            <span v-if="submitting"><i class="bi bi-arrow-repeat spin"></i> Starting…</span>
            <span v-else>Dub it</span>
          </button>
        </div>
      </div>

      <div class="dub-mode" role="group" aria-label="How the dub is made">
        <button type="button" :class="['mode-pill', { on: !fast }]" :aria-pressed="!fast" :disabled="!lipSyncReady" @click="fast = false">
          <strong><i class="bi bi-person-video3"></i> Lip-synced</strong><span>Best quality. About 12 min per minute of someone talking on screen, plus about 5 min of setup on a new video.</span>
        </button>
        <button type="button" :class="['mode-pill', { on: fast }]" :aria-pressed="fast" @click="fast = true">
          <strong><i class="bi bi-soundwave"></i> Voice only</strong><span>Fastest. About 6 min for a one-minute video; mouths stay as filmed. Lip sync can be added later.</span>
        </button>
      </div>
      <p v-if="!sourceJob" class="composer-hint">
        <template v-if="mode === 'url'"><template v-if="kind"><strong>{{ kind.label }}.</strong> </template>YouTube, Vimeo, news and social video, or a direct video file. Up to {{ config?.max_video_minutes }} minutes.</template>
        <template v-else>Your video is saved in this studio. Audio and text are processed by the configured AI providers.</template>
      </p>

      <div class="part-row">
        <label class="name-field"><i class="bi bi-tag" aria-hidden="true"></i><input v-model="title" type="text" maxlength="140" placeholder="Name this dub (optional)" aria-label="Name this dub" /></label>
        <label v-if="!sourceJob" class="switch"><input v-model="partOnly" type="checkbox" /><span></span><em>Dub only a part</em></label>
        <template v-if="partOnly && !sourceJob">
          <label class="part-time">from <input v-model="clipStart" type="text" inputmode="numeric" placeholder="m:ss" aria-label="Start of the part" /></label>
          <label class="part-time">to <input v-model="clipEnd" type="text" inputmode="numeric" placeholder="m:ss" aria-label="End of the part" /></label>
          <small class="field-hint">{{ partValid ? `${Math.max(0, Math.round(parseClock(clipEnd) - parseClock(clipStart)))} s of the video` : 'Use seconds, m:ss or h:mm:ss. Choose at least 1 second within the video limit.' }}</small>
        </template>
      </div>

      <div v-if="submitting && mode === 'file'" class="upload-progress" role="progressbar" aria-label="Video upload" :aria-valuenow="Math.round(uploadProgress * 100)" aria-valuemin="0" aria-valuemax="100">
        <div :style="{ width: Math.round(uploadProgress * 100) + '%' }"></div>
        <span>{{ uploadProgress < 1 ? `Uploading ${Math.round(uploadProgress * 100)}%` : 'Starting…' }}</span>
      </div>

      <details class="settings" :open="settingsOpen" @toggle="settingsOpen = $event.target.open">
        <summary><span><i class="bi bi-sliders2"></i> Voices, style and more</span><i class="bi bi-chevron-down caret"></i></summary>
        <div class="settings-grid">
          <div class="field">
            <span>Language spoken in the video</span>
            <OptionPicker v-model="sourceLanguage" :options="sourceLanguageOptions" label="Language spoken in the video" />
            <small class="field-hint">Telling us gives the cleanest transcript.</small>
          </div>
          <div class="field">
            <span>Voices</span>
            <OptionPicker v-model="voiceMode" :options="voiceModeOptions" label="Voices" />
            <small class="field-hint">{{ voiceMode === 'clone' ? 'Learned from how each person sounds in the original.' : 'Matched to how each person sounds. Change any of them after the dub.' }}</small>
          </div>
          <div class="field">
            <span>Delivery</span>
            <OptionPicker v-model="deliveryStyle" :options="deliveryOptions" label="Delivery" />
            <small class="field-hint">The overall tone. Adjust per speaker later.</small>
          </div>
          <div class="field">
            <span>Casting</span>
            <OptionPicker v-model="casting" :options="castingOptions" label="Casting" />
            <small class="field-hint">{{ casting === 'auto' ? 'Each speaker gets a voice for their register and the kind of video. You can change any of them afterwards.' : 'In the order people first speak. Speakers beyond your picks are matched automatically.' }}</small>
          </div>
          <template v-if="casting === 'manual'">
            <div v-for="n in slots" :key="n" class="field">
              <span>Speaker {{ n }}{{ n === 1 ? ' (first to speak)' : '' }}</span>
              <OptionPicker v-model="chosenVoices[n - 1]" :options="voiceOptions" :label="'Speaker ' + n" />
            </div>
          </template>
          <div class="field">
            <span>Lip sync</span>
            <OptionPicker v-model="lipModel" :options="lipOptions" label="Lip sync" />
          </div>
          <div class="field">
            <span>How many people speak</span>
            <OptionPicker v-model="numSpeakers" :options="speakerCountOptions" label="How many people speak" />
          </div>
          <div class="field">
            <span>Music and sound</span>
            <OptionPicker v-model="backgroundMode" :options="backgroundOptions" label="Music and sound" />
          </div>
        </div>
      </details>

      <p v-if="health && !ready" class="setup-warning"><i class="bi bi-exclamation-triangle"></i> {{ !health.bodhan_configured ? 'The voice services need to be configured before you can start.' : 'Video processing is unavailable. The studio needs FFmpeg.' }}</p>
    </form>


    <section v-if="recent.length" class="recent" aria-label="Recent dubs">
      <header class="recent-head"><h2>Recent dubs</h2></header>
      <div class="recent-grid">
        <button v-for="item in recent.slice(0, 6)" :key="item.id" :class="['recent-card', 'status-' + item.status]" @click="emit('open', item.id)">
          <span class="recent-thumb"><img v-if="item.poster_url" :src="item.poster_url" alt="" loading="lazy" /><i v-else class="bi bi-film"></i></span>
          <span class="recent-copy">
            <strong>{{ item.title || languageName(item.language) + ' dub' }}</strong>
            <small>{{ item.status === 'completed' ? 'Ready' : item.status === 'running' ? friendlyStage(item) : item.status === 'queued' ? 'Waiting to start' : item.status === 'cancelled' ? 'Stopped' : 'Needs attention' }} · {{ date(item.finished_at || item.updated_at || item.created_at) }}</small>
          </span>
        </button>
      </div>
    </section>
    <footer class="studio-copyright">
      <small>© {{ copyrightYear }} AvnishJ29 · Dubloom</small>
    </footer>
  </div>
</template>
