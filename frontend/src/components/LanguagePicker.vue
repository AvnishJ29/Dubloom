<script setup>
import { computed, nextTick, onBeforeUnmount, ref, useId, watch } from 'vue'
import { NATIVE_NAMES } from '../lib/format'

const props = defineProps({
  modelValue: String,
  languages: { type: Object, default: () => ({}) },
  label: { type: String, default: 'into' },
})
const emit = defineEmits(['update:modelValue'])
const listId = useId()
const trigger = ref(null)
const open = ref(false)
const query = ref('')
const cursor = ref(0)
const box = ref(null)
const input = ref(null)
const all = computed(() =>
  Object.entries(props.languages)
    .map(([code, name]) => ({ code, name, native: NATIVE_NAMES[code] || '' }))
    .sort((a, b) => a.name.localeCompare(b.name)),
)
const shown = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return all.value
  return all.value.filter(
    (item) =>
      item.name.toLowerCase().includes(q) ||
      item.native.includes(query.value.trim()) ||
      item.code === q,
  )
})
const current = computed(
  () =>
    all.value.find((item) => item.code === props.modelValue) || {
      code: props.modelValue,
      name: props.modelValue || 'Choose',
      native: '',
    },
)
function choose(item) {
  emit('update:modelValue', item.code)
  open.value = false
  query.value = ''
  trigger.value?.focus()
}
async function toggle() {
  open.value = !open.value
  if (open.value) {
    await nextTick()
    input.value?.focus()
    cursor.value = Math.max(
      0,
      shown.value.findIndex((i) => i.code === props.modelValue),
    )
  }
}
function onKey(event) {
  if (event.key === 'Escape') {
    open.value = false
    trigger.value?.focus()
    event.preventDefault()
    return
  }
  if (event.key === 'ArrowDown') {
    cursor.value = Math.min(shown.value.length - 1, cursor.value + 1)
    event.preventDefault()
  }
  if (event.key === 'ArrowUp') {
    cursor.value = Math.max(0, cursor.value - 1)
    event.preventDefault()
  }
  if (['ArrowDown', 'ArrowUp'].includes(event.key))
    nextTick(() =>
      document.getElementById(listId + '-' + cursor.value)?.scrollIntoView({ block: 'nearest' }),
    )
  if (event.key === 'Enter' && shown.value[cursor.value]) {
    choose(shown.value[cursor.value])
    event.preventDefault()
  }
}
watch(query, () => {
  cursor.value = 0
})
function away(event) {
  if (open.value && box.value && !box.value.contains(event.target)) open.value = false
}
watch(open, (value) => {
  value
    ? document.addEventListener('mousedown', away)
    : document.removeEventListener('mousedown', away)
})
onBeforeUnmount(() => document.removeEventListener('mousedown', away))
</script>

<template>
  <div ref="box" class="lang-picker" :class="{ open }" @focusout="event => { if (!box?.contains(event.relatedTarget)) open = false }">
    <span class="lang-label">{{ label }}</span>
    <button ref="trigger" type="button" class="lang-current" :aria-controls="listId" :aria-expanded="open" aria-haspopup="dialog" @click="toggle">
      <span class="lang-native">{{ current.native || current.name }}</span>
      <span class="lang-english">{{ current.native ? current.name : '' }}</span>
      <i class="bi bi-chevron-down"></i>
    </button>
    <div v-if="open" class="lang-pop" role="dialog" aria-label="Choose a language">
      <label class="lang-search">
        <i class="bi bi-search"></i>
        <input ref="input" v-model="query" type="search" :placeholder="`Search ${all.length} languages…`" aria-label="Search languages" role="combobox" aria-autocomplete="list" :aria-expanded="open" :aria-controls="listId" :aria-activedescendant="shown[cursor] ? listId + '-' + cursor : undefined" @keydown="onKey" />
      </label>
      <ul :id="listId" class="lang-grid" role="listbox" aria-label="Languages">
        <li v-for="(item, i) in shown" :key="item.code" :id="listId + '-' + i" role="option" @mousedown.prevent :aria-selected="item.code === modelValue"
            :class="{ on: item.code === modelValue, cursor: i === cursor }" @click="choose(item)" @mousemove="cursor = i">
          <span class="lang-native">{{ item.native || item.name }}</span>
          <span class="lang-english">{{ item.name }}</span>
        </li>
        <li v-if="!shown.length" class="lang-empty">No language matches "{{ query }}"</li>
      </ul>
      <small class="lang-foot">{{ all.length }} languages · voices, script and lip sync for every one of them</small>
    </div>
  </div>
</template>
