<script setup>
import { computed, reactive, watch } from 'vue'
import { durationLabel, speakerLabel, speakerOrder, timecode } from '../lib/format'

const props = defineProps({
  job: Object,
  voices: { type: Array, default: () => [] },
  colors: Object,
  editable: Boolean,
  busy: Boolean,
  styles: { type: Array, default: () => [] },
})
const emit = defineEmits(['revoice', 'seek', 'merge'])
const draft = reactive({})
const styleDraft = reactive({})
const speakers = computed(() => speakerOrder(props.job))
const currentStyle = (id) => props.job.speaker_styles?.[id] || props.job.delivery_style || 'neutral'
watch(
  () => props.job?.speaker_voices,
  (voices) => {
    for (const id in voices || {}) {
      draft[id] = voices[id]
      styleDraft[id] = currentStyle(id)
    }
  },
  { immediate: true, deep: true },
)
const changes = computed(() =>
  Object.fromEntries(
    speakers.value
      .filter((id) => draft[id] && draft[id] !== props.job.speaker_voices?.[id])
      .map((id) => [id, draft[id]]),
  ),
)
const styleChanges = computed(() =>
  Object.fromEntries(
    speakers.value
      .filter((id) => styleDraft[id] && styleDraft[id] !== currentStyle(id))
      .map((id) => [id, styleDraft[id]]),
  ),
)
const dirty = computed(
  () => Object.keys(changes.value).length + Object.keys(styleChanges.value).length > 0,
)
const voiceWord = (gender) =>
  gender === 'female' ? 'Female voice' : gender === 'male' ? 'Male voice' : 'Voice'
const voiceCard = (name) => props.voices.find((v) => v.name === name)
const confidence = (id) =>
  Math.round((props.job.speaker_profiles?.[id]?.gender_confidence || 0) * 100)
function flipGender(id) {
  // the best voice of the other gender, in library order; Apply re-renders this speaker's lines with it
  const current = voiceCard(draft[id])
  const other = current?.gender === 'female' ? 'male' : 'female'
  const used = new Set(Object.values(draft))
  const pick =
    props.voices.find((v) => v.gender === other && !used.has(v.name)) ||
    props.voices.find((v) => v.gender === other)
  if (pick) draft[id] = pick.name
}
function reset() {
  for (const id of speakers.value) {
    draft[id] = props.job.speaker_voices?.[id]
    styleDraft[id] = currentStyle(id)
  }
}
</script>

<template>
  <section class="panel speakers">
    <header class="panel-head">
      <div><h2>Speakers</h2><span class="panel-sub">{{ speakers.length }} found{{ editable ? ' · change a voice or tone, then re-render' : '' }}</span></div>
      <div v-if="editable" class="panel-tools">
        <button v-if="dirty" class="button ghost small" :disabled="busy" @click="reset">Undo</button>
        <button class="button primary small" :disabled="!dirty || busy" @click="emit('revoice', { speaker_voices: changes, speaker_styles: styleChanges })"><i class="bi bi-arrow-repeat"></i> Apply and re-render</button>
      </div>
    </header>
    <div class="speaker-grid">
      <article v-for="(id, index) in speakers" :key="id" class="speaker-card" :style="{ '--speaker': colors[id] }">
        <div class="speaker-top">
          <span class="speaker-avatar">{{ index + 1 }}</span>
          <div class="speaker-copy">
            <strong>{{ speakerLabel(id, job.speaker_profiles) }}</strong>
            <small v-if="job.speaker_profiles?.[id]">
              {{ voiceWord(job.speaker_profiles[id].gender) }}<template v-if="confidence(id) && confidence(id) < 90"> ({{ confidence(id) }}% sure)</template> · speaks for {{ durationLabel(job.speaker_profiles[id].speech_seconds) }} · {{ job.speaker_profiles[id].turns }} line{{ job.speaker_profiles[id].turns === 1 ? '' : 's' }}
              <template v-if="job.speaker_profiles[id].split_from"> · separated from {{ speakerLabel(job.speaker_profiles[id].split_from, job.speaker_profiles) }} by voice</template>
            </small>
            <button v-if="editable && speakers.length > 1" type="button" class="link-button" :disabled="busy"
                    @click="emit('merge', { speaker: id, into: job.speaker_profiles?.[id]?.split_from && job.speaker_voices?.[job.speaker_profiles[id].split_from] ? job.speaker_profiles[id].split_from : speakers.find(other => other !== id) })">
              Same person as {{ speakerLabel(job.speaker_profiles?.[id]?.split_from && job.speaker_voices?.[job.speaker_profiles[id].split_from] ? job.speaker_profiles[id].split_from : speakers.find(other => other !== id), job.speaker_profiles) }}? Merge
            </button>
          </div>
          <button v-if="job.speaker_profiles?.[id]" class="icon-button" :title="'Jump to ' + timecode(job.speaker_profiles[id].first_seen)" :aria-label="'Jump to first line at ' + timecode(job.speaker_profiles[id].first_seen)" @click="emit('seek', job.speaker_profiles[id].first_seen)"><i class="bi bi-skip-start"></i></button>
        </div>
        <label class="field compact">
          <span>Voice</span>
          <select v-if="editable" v-model="draft[id]" :disabled="busy">
            <option v-for="voice in voices" :key="voice.name" :value="voice.name">{{ voice.name }} · {{ voice.gender }}<template v-if="voice.language"> · {{ voice.language }}</template><template v-if="voice.best_for?.length"> · {{ voice.best_for[0] }}</template></option>
          </select>
          <output v-else>{{ job.speaker_voices?.[id] || 'Chosen automatically' }}</output>
          <small v-if="voiceCard(draft[id])?.description" class="field-hint">{{ voiceCard(draft[id]).description }}</small>
          <small v-if="job.voice_reasons?.[id] && draft[id] === job.speaker_voices?.[id]" class="field-hint why">Cast because: {{ job.voice_reasons[id].split(' · ').pop() }}</small>
          <button v-if="editable" type="button" class="link-button" :disabled="busy" @click="flipGender(id)">Sounds like a {{ voiceCard(draft[id])?.gender === 'female' ? 'man' : 'woman' }}? Switch</button>
        </label>
        <label class="field compact">
          <span>Tone</span>
          <select v-if="editable" v-model="styleDraft[id]" :disabled="busy">
            <option v-for="item in styles" :key="item.id" :value="item.id">{{ item.label }}</option>
          </select>
          <output v-else>{{ (styles.find(s => s.id === currentStyle(id)) || {}).label || currentStyle(id) }}</output>
        </label>
      </article>
    </div>
  </section>
</template>
