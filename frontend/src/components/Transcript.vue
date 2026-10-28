<script setup>
import { computed, reactive, ref, watch } from 'vue'
const STATE_LABEL = {
  heard: 'heard',
  translated: 'translated',
  voiced: 'voiced',
  'lip-synced': 'lip-synced',
}
const STATE_ICON = {
  heard: 'bi bi-ear',
  translated: 'bi bi-translate',
  voiced: 'bi bi-mic',
  'lip-synced': 'bi bi-person-video',
}
const STATE_TITLE = {
  heard: 'Transcribed; translation next',
  translated: 'Translated and enhanced; voice next',
  voiced: 'Voiced and placed; lip sync next',
  'lip-synced': 'Lips synced in the picture',
}
import { timecode, speakerLabel, dateTime } from '../lib/format'
import { diffWords } from '../lib/diff'
import { paces, fitLine, nextStarts, timingIssue } from '../lib/timing'

const props = defineProps({
  segments: { type: Array, default: () => [] },
  colors: { type: Object, default: () => ({}) },
  voices: { type: Object, default: () => ({}) },
  reviewed: { type: Boolean, default: false },
  profiles: { type: Object, default: () => ({}) },
  activeIndex: { type: Number, default: -1 },
  canSeek: Boolean,
  editable: Boolean,
  busy: Boolean,
  languageName: { type: String, default: '' },
  proposals: { type: Object, default: null },
  rewriting: Boolean,
  jobId: { type: String, default: '' },
  styles: { type: Array, default: () => [] },
  takes: { type: Object, default: () => ({}) },
  taking: Boolean,
})
const emit = defineEmits(['seek', 'apply', 'rewrite', 'take', 'choose'])
const directing = ref(-1) // index of the line whose director panel is open
const inline = ref(null) // { index, field, value } while one row of one line is being edited in place
function editRow(row, which) {
  inline.value = { index: row.index, field: which, value: row[which] || '' }
}
function saveRow() {
  const { index, field, value } = inline.value
  const current = props.segments[index]?.[field] || ''
  if (value.trim() && value.trim() !== current.trim())
    emit('apply', { lines: [{ index, [field]: value.trim() }] })
  inline.value = null
}
const takeStyle = ref('neutral')
const takeText = ref('')
function direct(row) {
  if (directing.value === row.index) {
    directing.value = -1
    return
  }
  directing.value = row.index
  takeStyle.value = row.style_override || 'neutral'
  takeText.value = row.translation || ''
}
const pace = (take) => {
  const r = take.slot_seconds ? take.seconds / take.slot_seconds : 1
  return r > 1.15 ? 'fast for the slot' : r < 0.85 ? 'short for the slot' : 'fits the slot'
}
const showOriginal = ref(true)
const rates = computed(() => paces(props.segments))
const nexts = computed(() => nextStarts(props.segments))
const fit = (index, text) => fitLine(props.segments, rates.value, index, text, nexts.value) // live timing of a line being edited
const changedWords = (earlier, now) =>
  diffWords(earlier, now || '').filter((part) => part.op !== 'ins') // an earlier version, its differing words marked
const query = ref('')
const editing = ref(false)
const drafts = reactive({})
const instruction = ref('')
const proposalNote = ref('')
const aiDrafts = reactive({}) // index -> the AI's proposed text, so an applied proposal is tagged as AI enhanced
const compared = reactive({}) // index -> { beforeText, afterText, before: [parts], after: [parts] } for each line the AI proposes to change
const showingHistory = ref(-1) // index of the line whose earlier versions are open
const BY_LABEL = {
  bodhan: 'Bodhan translation',
  editor: 'enhanced for this content',
  you: 'your edit',
  ai: 'AI suggestion you applied',
}
const byLabel = (entry) =>
  entry.field === 'text'
    ? entry.by === 'you'
      ? 'your correction'
      : 'as recognised'
    : BY_LABEL[entry.by] || entry.by
function draftState(index) {
  const c = compared[index]
  if (!c) return ''
  const d = (drafts[index] || '').trim()
  return d === c.afterText ? 'suggestion' : d === c.beforeText ? 'current' : 'yours'
}
function clearProposal() {
  for (const key of Object.keys(aiDrafts)) delete aiDrafts[key]
  for (const key of Object.keys(compared)) delete compared[key]
}
const field = ref('translation') // what the editor and the AI work on: the dub's lines or the original transcript
const fieldLabel = computed(() => (field.value === 'text' ? 'original' : languageNameOrDub()))
function languageNameOrDub() {
  return props.languageName || 'dub'
}
function switchField(next) {
  if (next !== field.value) {
    field.value = next
    clearProposal()
    proposalNote.value = ''
    resetDrafts()
  }
} // a proposal belongs to the field it was made for

// Proposals from the AI land in the editor as drafts: the user reads, edits and applies only what they keep.
watch(
  () => props.proposals,
  (value) => {
    clearProposal()
    if (!value) return
    field.value = value.field || 'translation'
    resetDrafts()
    for (const line of value.lines || []) {
      drafts[line.index] = line.after
      aiDrafts[line.index] = line.after
      const parts = diffWords(line.before || '', line.after || '')
      compared[line.index] = {
        beforeText: (line.before || '').trim(),
        afterText: (line.after || '').trim(),
        before: parts.filter((p) => p.op !== 'ins'),
        after: parts.filter((p) => p.op !== 'del'),
      }
    }
    editing.value = true
    proposalNote.value = value.lines?.length
      ? `The AI suggests changes to ${value.lines.length} line${value.lines.length === 1 ? '' : 's'} for “${value.instruction}”. Each shows the current words and the suggestion: keep either, or edit, then apply. Nothing is lost: a changed line keeps its earlier versions.`
      : `The AI found nothing to change for “${value.instruction}”.`
  },
)
function askAi() {
  const text = instruction.value.trim()
  if (text.length < 3 || props.rewriting) return
  const scope = query.value ? rows.value.map((row) => row.index) : null // a search narrows what may change
  emit('rewrite', { instruction: text, lines: scope, field: field.value })
}

watch(
  () => props.segments,
  () => {
    if (!editing.value) resetDrafts()
  },
  { immediate: true, deep: true },
)
function resetDrafts() {
  for (const key of Object.keys(drafts)) delete drafts[key]
  props.segments.forEach((segment, index) => {
    drafts[index] = segment[field.value] || ''
  })
}
const changed = computed(() =>
  props.segments
    .map((segment, index) => index)
    .filter(
      (index) => (drafts[index] || '').trim() !== (props.segments[index][field.value] || '').trim(),
    ),
)
const rows = computed(() =>
  props.segments
    .map((segment, index) => ({ ...segment, index }))
    .filter(
      (row) =>
        !query.value ||
        (row.translation + ' ' + row.text + ' ' + (drafts[row.index] || ''))
          .toLowerCase()
          .includes(query.value.toLowerCase()),
    ),
)
function startEditing() {
  resetDrafts()
  editing.value = true
}
function cancelEditing() {
  editing.value = false
  proposalNote.value = ''
  field.value = 'translation'
  clearProposal()
  resetDrafts()
}
function apply() {
  const lines = changed.value
    .map((index) => ({
      index,
      [field.value]: drafts[index].trim(),
      via: aiDrafts[index] && aiDrafts[index].trim() === drafts[index].trim() ? 'ai' : undefined,
    }))
    .filter((line) => line[field.value])
  if (!lines.length) return
  emit('apply', { lines })
  editing.value = false
  proposalNote.value = ''
  instruction.value = ''
  clearProposal()
}
</script>

<template>
  <section class="panel transcript">
    <header class="panel-head">
      <div><h2>{{ editing ? 'Edit the script' : 'Script' }}</h2><span class="panel-sub">{{ segments.length }} lines · {{ editing ? (field === 'text' ? 'corrected original lines are translated and voiced again' : 'change any line, then apply') : 'original as recognised, translated by Bodhan, enhanced for this content · a pencil changes one line on its own · a changed line keeps every earlier version' }}</span></div>
      <div class="panel-tools">
        <div v-if="editable" class="composer-tabs small" role="tablist" aria-label="What to edit">
          <button type="button" role="tab" :aria-selected="field === 'translation'" :class="{ active: field === 'translation' }" :disabled="busy" @click="switchField('translation')">{{ languageName || 'Dub' }}</button>
          <button type="button" role="tab" :aria-selected="field === 'text'" :class="{ active: field === 'text' }" :disabled="busy" @click="switchField('text')">Original</button>
        </div>
        <label v-if="!editing" class="search"><i class="bi bi-search"></i><input v-model="query" type="search" placeholder="Find a line" aria-label="Find a line" /></label>
        <label v-if="!editing" class="switch"><input v-model="showOriginal" type="checkbox" /><span></span><em>Show original and translated rows</em></label>
        <button v-if="editable && !editing" class="button ghost small" :disabled="busy" @click="startEditing"><i class="bi bi-pencil-square"></i> Edit script</button>
        <template v-if="editing">
          <button class="button ghost small" :disabled="busy" @click="cancelEditing">Discard</button>
          <button class="button primary small" :disabled="busy || !changed.length" @click="apply"><i class="bi bi-arrow-repeat"></i> Apply {{ changed.length || '' }} change{{ changed.length === 1 ? '' : 's' }}</button>
        </template>
      </div>
    </header>
    <form v-if="editable && !editing" class="ask-ai" @submit.prevent="askAi">
      <i class="bi bi-stars" aria-hidden="true"></i>
      <input v-model="instruction" type="text" maxlength="600" :disabled="rewriting || busy" aria-label="Tell the AI how to change the script"
             :placeholder="field === 'text' ? 'Tell the AI how to correct the original transcript: e.g. the name is Mitesh, write formulas as spoken' : query ? `Change the ${rows.length} matching line${rows.length === 1 ? '' : 's'}: e.g. make them more formal` : 'Tell the AI how to change the dub: e.g. make it more formal, use simpler words, keep names in English'" />
      <button class="button primary small" type="submit" :disabled="instruction.trim().length < 3 || rewriting || busy">
        <span v-if="rewriting"><i class="bi bi-arrow-repeat spin"></i> Thinking…</span><span v-else>Propose changes</span>
      </button>
    </form>
    <p v-if="editing" class="tone-bar"><small class="tone-note">{{ proposalNote || 'Only the lines you change are re-voiced and re-synced. Everything else stays as it is.' }}</small></p>
    <p v-if="!rows.length" class="empty">No line matches that.</p>
    <ol class="lines">
      <li v-for="row in rows" :key="row.index" :class="['line', { active: row.index === activeIndex, changed: editing && changed.includes(row.index) }]" :style="{ '--speaker': colors[row.speaker] || 'var(--brand)' }">
        <button class="line-time" :disabled="!canSeek" @click="emit('seek', row.dub_start ?? row.start)">{{ timecode(row.start) }}</button>
        <div class="line-body">
          <span class="line-speaker"><i class="dot"></i>{{ speakerLabel(row.speaker, profiles) }}<span v-if="voices?.[row.speaker]" class="line-voice" :title="'Voice used for this speaker'">· voiced by {{ voices[row.speaker] }}</span>
            <em v-if="row.edited && !editing" class="edited-badge" :class="{ enhanced: row.ai_edited }" :title="row.ai_edited ? 'You asked the AI to rewrite this line' : 'You edited this line'"><i :class="row.ai_edited ? 'bi bi-stars' : 'bi bi-pencil'"></i> {{ row.ai_edited ? 'AI enhanced' : 'edited' }}</em>
            <em v-if="row.source_edited && !editing" class="edited-badge" title="You corrected the original line; it was translated again"><i class="bi bi-pencil"></i> original corrected</em>
            <button v-if="!editing && timingIssue(row)" type="button" :class="['edited-badge', 'timing', timingIssue(row).kind]" :disabled="!editable || busy" :title="timingIssue(row).text + (editable ? ' Click to shorten it: the meter shows when it fits.' : '')" @click="editRow(row, 'translation')"><i class="bi bi-speedometer2"></i> {{ timingIssue(row).short }}</button>
            <em v-if="row.take && !editing" class="edited-badge enhanced" title="You picked a take for this line"><i class="bi bi-mic"></i> {{ (styles.find(s => s.id === row.style_override) || {}).label || row.style_override || 'take' }}</em>
            <button v-if="row.history?.length && !editing" type="button" class="direct-button" :class="{ open: showingHistory === row.index }" title="Earlier versions of this line. Nothing is deleted; any of them can be restored." @click="showingHistory = showingHistory === row.index ? -1 : row.index"><i class="bi bi-clock-history"></i> {{ row.history.length }} earlier</button>
            <button v-if="editable && !editing" type="button" class="direct-button" :class="{ open: directing === row.index }" :title="'Direct this line: try a delivery, pick a take'" @click="direct(row)"><i class="bi bi-sliders2-vertical"></i> Direct</button></span>
          <div v-if="editing && compared[row.index]" class="compare">
            <div class="compare-row"><em class="tag">now</em><p><template v-for="(part, i) in compared[row.index].before" :key="i"><del v-if="part.op === 'del'">{{ part.text }}</del><template v-else>{{ part.text }}</template></template></p></div>
            <div class="compare-row"><em class="tag brand">suggested</em><p><template v-for="(part, i) in compared[row.index].after" :key="i"><ins v-if="part.op === 'ins'">{{ part.text }}</ins><template v-else>{{ part.text }}</template></template></p></div>
            <div class="compare-tools">
              <button type="button" :class="['chip-button', { active: draftState(row.index) === 'suggestion' }]" :disabled="busy" @click="drafts[row.index] = compared[row.index].afterText">Use suggestion</button>
              <button type="button" :class="['chip-button', { active: draftState(row.index) === 'current' }]" :disabled="busy" @click="drafts[row.index] = compared[row.index].beforeText">Keep current</button>
              <small v-if="draftState(row.index) === 'yours'" class="note">your own wording below will be used</small>
            </div>
          </div>
          <textarea v-if="editing" v-model="drafts[row.index]" rows="2" :lang="field === 'text' ? undefined : row.language" :disabled="busy"></textarea>
          <small v-if="editing && field === 'translation' && fit(row.index, drafts[row.index])" :class="['fit-meter', fit(row.index, drafts[row.index]).kind]">{{ fit(row.index, drafts[row.index]).text }}</small>
          <small v-if="editing && field === 'text' && row.translation">{{ row.translation }}</small>
          <div v-if="!editing" class="script-rows">
            <div v-if="showOriginal && row.text" :class="['script-row', { editing: inline?.index === row.index && inline.field === 'text' }]">
              <em class="tag">original</em>
              <template v-if="inline?.index === row.index && inline.field === 'text'">
                <textarea v-model="inline.value" rows="2" :disabled="busy" @keydown.esc="inline = null"></textarea>
                <span class="row-tools"><button type="button" class="button primary small" :disabled="busy" @click="saveRow">Save, translate and voice this line</button><button type="button" class="button ghost small" @click="inline = null">Cancel</button></span>
              </template>
              <template v-else>
                <p class="muted-text">{{ row.text }}</p>
                <button v-if="editable" type="button" class="icon-button tiny" title="Correct the original words of this line" aria-label="Edit the original line" @click="editRow(row, 'text')"><i class="bi bi-pencil"></i></button>
              </template>
            </div>
            <div v-if="showOriginal && row.translation" class="script-row">
              <em class="tag">translated</em>
              <p class="muted-text">{{ row.machine_translation || row.translation }}<small v-if="!row.edited && (!row.machine_translation || row.machine_translation === row.translation)" class="note"> · kept by the editor</small></p>
            </div>
            <div v-if="row.translation" :class="['script-row', 'enhanced', { editing: inline?.index === row.index && inline.field === 'translation' }]">
              <em class="tag brand" :title="row.machine_translation && row.machine_translation !== row.translation ? 'Improved by the script editor for this content' : 'What the voice says'">{{ row.edited ? 'yours' : 'enhanced' }}</em>
              <template v-if="inline?.index === row.index && inline.field === 'translation'">
                <textarea v-model="inline.value" rows="2" :lang="row.language" :disabled="busy" @keydown.esc="inline = null"></textarea>
                <small v-if="fit(row.index, inline.value)" :class="['fit-meter', fit(row.index, inline.value).kind]">{{ fit(row.index, inline.value).text }}</small>
                <span class="row-tools"><button type="button" class="button primary small" :disabled="busy" @click="saveRow">Save and voice this line</button><button type="button" class="button ghost small" @click="inline = null">Cancel</button></span>
              </template>
              <template v-else>
                <p>{{ row.translation }}</p>
                <button v-if="editable" type="button" class="icon-button tiny" title="Change this line; only this line is voiced again" aria-label="Edit this line" @click="editRow(row, 'translation')"><i class="bi bi-pencil"></i></button>
              </template>
            </div>
            <p v-if="!row.translation" class="muted-text">{{ row.text }}</p>
          </div>
          <ol v-if="showingHistory === row.index && !editing" class="line-history" aria-label="Earlier versions of this line">
            <li v-for="(entry, i) in [...row.history].reverse()" :key="i">
              <small>{{ byLabel(entry) }} · {{ entry.field === 'text' ? 'original' : (languageName || 'dub') }} · replaced {{ dateTime(entry.at) }}</small>
              <p><template v-for="(part, k) in changedWords(entry.text, row[entry.field])" :key="k"><mark v-if="part.op === 'del'">{{ part.text }}</mark><template v-else>{{ part.text }}</template></template></p>
              <button v-if="editable" type="button" class="button ghost small" :disabled="busy" :title="entry.field === 'text' ? 'Put these original words back; the line is translated and voiced again' : 'Put these words back; only this line is voiced again'" @click="emit('apply', { lines: [{ index: row.index, [entry.field]: entry.text }] })"><i class="bi bi-arrow-counterclockwise"></i> Restore</button>
            </li>
          </ol>
          <div v-if="directing === row.index && !editing" class="director">
            <div class="chips" role="radiogroup" aria-label="Delivery">
              <button v-for="item in styles" :key="item.id" type="button" role="radio" :aria-checked="takeStyle === item.id" :class="['chip-button', { active: takeStyle === item.id }]" @click="takeStyle = item.id">{{ item.label }}</button>
            </div>
            <div class="director-row">
              <input v-model="takeText" type="text" maxlength="2000" :lang="row.language" aria-label="Words for this take" :disabled="taking" />
              <button type="button" class="button primary small" :disabled="taking || !takeText.trim()" @click="emit('take', { index: row.index, style: takeStyle, text: takeText.trim() })">
                <span v-if="taking"><i class="bi bi-arrow-repeat spin"></i> Voicing…</span><span v-else><i class="bi bi-mic"></i> Generate take</span>
              </button>
            </div>
            <ol v-if="(takes[row.index] || []).length" class="takes">
              <li v-for="take in takes[row.index]" :key="take.number" :class="['take', { chosen: row.take === take.file }]">
                <audio :src="`/api/jobs/${jobId}/takes/${take.file}`" controls preload="none"></audio>
                <div class="take-copy"><strong>Take {{ take.number }} · {{ (styles.find(s => s.id === take.style) || {}).label || take.style }}</strong><small>{{ take.seconds }} s · {{ pace(take) }}<template v-if="take.text !== row.translation"> · new words</template></small></div>
                <button type="button" class="button ghost small" :disabled="busy || row.take === take.file" @click="emit('choose', { index: row.index, number: take.number })">{{ row.take === take.file ? 'In use' : 'Use this take' }}</button>
              </li>
            </ol>
            <p v-else class="field-hint">Pick a delivery, change the words if you like, and generate a take to hear it. Nothing is re-rendered until you choose one.</p>
          </div>
        </div>
      </li>
    </ol>
  </section>
</template>
