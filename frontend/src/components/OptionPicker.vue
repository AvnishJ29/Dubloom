<script setup>
import { computed, nextTick, onBeforeUnmount, ref, useId, watch } from 'vue'

const props = defineProps({
  modelValue: [String, Number, Boolean],
  options: { type: Array, default: () => [] },
  label: { type: String, required: true },
})
const emit = defineEmits(['update:modelValue'])
const id = useId()
const box = ref(null)
const trigger = ref(null)
const search = ref(null)
const open = ref(false)
const query = ref('')
const cursor = ref(-1)
const selected = computed(
  () =>
    props.options.find((option) => option.value === props.modelValue) ||
    (props.modelValue == null ? props.options.find((option) => option.value === '') : undefined),
)
const shown = computed(() => {
  const needle = query.value.trim().toLowerCase()
  return props.options.filter((option) => option.label.toLowerCase().includes(needle))
})

function close(restoreFocus = false) {
  open.value = false
  query.value = ''
  if (restoreFocus) trigger.value?.focus()
}
function choose(option) {
  if (option.disabled) return
  emit('update:modelValue', option.value)
  close(true)
}
async function toggle() {
  if (open.value) {
    close()
    return
  }
  open.value = true
  cursor.value = shown.value.findIndex(
    (option) => option.value === props.modelValue && !option.disabled,
  )
  if (cursor.value < 0) cursor.value = shown.value.findIndex((option) => !option.disabled)
  await nextTick()
  search.value?.focus()
}
async function move(direction) {
  const count = shown.value.length
  if (!count) return
  let candidate = cursor.value
  for (let attempt = 0; attempt < count; attempt++) {
    candidate = (candidate + direction + count) % count
    if (!shown.value[candidate].disabled) {
      cursor.value = candidate
      await nextTick()
      document.getElementById(id + '-option-' + candidate)?.scrollIntoView({ block: 'nearest' })
      return
    }
  }
}
function onKey(event) {
  if (event.key === 'Escape') {
    event.preventDefault()
    close(true)
  } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault()
    move(event.key === 'ArrowDown' ? 1 : -1)
  } else if (event.key === 'Enter') {
    event.preventDefault()
    if (shown.value[cursor.value]) choose(shown.value[cursor.value])
  }
}
function away(event) {
  if (!box.value?.contains(event.target)) close()
}
watch(query, () => {
  cursor.value = shown.value.findIndex((option) => !option.disabled)
})
watch(open, (value) => {
  if (value) document.addEventListener('pointerdown', away)
  else document.removeEventListener('pointerdown', away)
})
onBeforeUnmount(() => document.removeEventListener('pointerdown', away))
</script>

<template>
  <div ref="box" class="lang-picker option-picker" :class="{ open }"
       @focusout="event => { if (!box?.contains(event.relatedTarget)) close() }">
    <button ref="trigger" type="button" class="lang-current" :aria-label="label + ': ' + (selected?.label || 'Choose')"
            aria-haspopup="dialog" :aria-expanded="open" :aria-controls="id + '-panel'" @click="toggle">
      <span class="option-label">{{ selected?.label || 'Choose' }}</span>
      <i class="bi bi-chevron-down" aria-hidden="true"></i>
    </button>
    <div v-if="open" :id="id + '-panel'" class="lang-pop" role="dialog" :aria-label="'Choose ' + label.toLowerCase()">
      <label class="lang-search">
        <i class="bi bi-search" aria-hidden="true"></i>
        <input ref="search" v-model="query" type="search" :aria-label="'Search ' + label.toLowerCase()"
               placeholder="Search options…" role="combobox" aria-autocomplete="list" aria-expanded="true"
               :aria-controls="id + '-list'" :aria-activedescendant="cursor >= 0 ? id + '-option-' + cursor : undefined"
               @keydown="onKey" />
      </label>
      <ul :id="id + '-list'" class="lang-grid option-list" role="listbox" :aria-label="label">
        <li v-for="(option, index) in shown" :id="id + '-option-' + index" :key="option.value"
            role="option" :aria-selected="option.value === modelValue" :aria-disabled="!!option.disabled"
            :class="{ on: option.value === modelValue, cursor: index === cursor, disabled: option.disabled }"
            @mousedown.prevent @click="choose(option)" @mousemove="cursor = option.disabled ? cursor : index">
          <span>{{ option.label }}</span>
          <i v-if="option.value === modelValue" class="bi bi-check2" aria-hidden="true"></i>
        </li>
        <li v-if="!shown.length" class="lang-empty" role="presentation">No matching options</li>
      </ul>
    </div>
  </div>
</template>
