<script setup>
import { computed, ref, watch } from 'vue'
import { timecode } from '../lib/format'

const props = defineProps({ job: Object, playerTime: { type: Number, default: 0 }, busy: Boolean })
const emit = defineEmits(['create', 'remove', 'seek'])
const start = ref(0)
const end = ref(0)
const layout = ref('face')
const subtitles = ref(false)
const duration = computed(() => props.job.duration || 0)
const length = computed(() => Math.max(0, end.value - start.value))
const valid = computed(
  () => length.value >= 1 && length.value <= 180 && end.value <= duration.value + 0.01,
)
const shorts = computed(() => [...(props.job.shorts || [])].reverse())
watch(
  () => props.job.id,
  () => {
    start.value = 0
    end.value = Math.min(duration.value, 30)
  },
  { immediate: true },
)
function useStart() {
  start.value = Math.round(props.playerTime * 10) / 10
  if (end.value <= start.value) end.value = Math.min(duration.value, start.value + 30)
}
function useEnd() {
  end.value = Math.round(props.playerTime * 10) / 10
}
const parse = (value) => {
  const [m, s] = String(value).split(':')
  return s === undefined ? Number(m) || 0 : (Number(m) || 0) * 60 + (Number(s) || 0)
}
</script>

<template>
  <section class="panel shorts">
    <header class="panel-head">
      <div><h2>Shorts</h2><span class="panel-sub">Cut a vertical clip of the dub, up to 3 minutes long.</span></div>
    </header>
    <div class="short-form">
      <label class="field compact"><span>From</span>
        <div class="time-row"><input :value="timecode(start)" @change="start = parse($event.target.value)" inputmode="numeric" aria-label="Start time" />
          <button type="button" class="button ghost small" @click="useStart" title="Use the player's current time">Now</button></div>
      </label>
      <label class="field compact"><span>To</span>
        <div class="time-row"><input :value="timecode(end)" @change="end = parse($event.target.value)" inputmode="numeric" aria-label="End time" />
          <button type="button" class="button ghost small" @click="useEnd" title="Use the player's current time">Now</button></div>
      </label>
      <label class="field compact"><span>Framing</span>
        <select v-model="layout">
          <option value="face">Follow the speaking face</option>
          <option value="centre">Centre of the frame</option>
          <option value="blur">Whole frame on blurred bars</option>
        </select>
      </label>
      <label class="switch tall"><input v-model="subtitles" type="checkbox" /><span></span><em>Burn in the dubbed lines</em></label>
      <button class="button primary" :disabled="!valid || busy" @click="emit('create', { start, end, layout, subtitles })">
        <span v-if="busy"><i class="bi bi-arrow-repeat spin"></i> Cutting…</span><span v-else>Create short · {{ Math.round(length) }} s</span>
      </button>
    </div>
    <p v-if="!valid && length" class="field-hint">A short is between 1 s and 3 min, inside the video.</p>
    <div v-if="shorts.length" class="short-grid">
      <article v-for="short in shorts" :key="short.number" class="short-card">
        <video :src="`/api/jobs/${job.id}/shorts/${short.number}`" controls playsinline preload="metadata"></video>
        <div class="short-copy">
          <strong>Short {{ short.number }}</strong>
          <small>{{ timecode(short.start) }}–{{ timecode(short.end) }} · {{ Math.round(short.end - short.start) }} s · {{ { face: 'follows the face', centre: 'centred', blur: 'blurred bars' }[short.layout] }}{{ short.subtitles ? ' · subtitled' : '' }}</small>
          <div class="short-tools">
            <a class="button ghost small" :href="`/api/jobs/${job.id}/shorts/${short.number}?download=true`" download><i class="bi bi-download"></i> Download</a>
            <button class="icon-button danger" :disabled="busy" :aria-label="'Delete short ' + short.number" @click="emit('remove', short.number)"><i class="bi bi-trash3"></i></button>
          </div>
        </div>
      </article>
    </div>
  </section>
</template>
