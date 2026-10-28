<script setup>
import { computed } from 'vue'
import { STAGES, durationLabel } from '../lib/format'

const props = defineProps({
  stageKey: String,
  status: String,
  stageSeconds: { type: Object, default: () => ({}) },
})
const currentIndex = computed(() => STAGES.findIndex((stage) => stage.key === props.stageKey))
function state(index) {
  if (props.status === 'completed') return 'done'
  if (currentIndex.value === -1)
    return props.status === 'queued'
      ? 'pending'
      : props.stageSeconds[STAGES[index].key]
        ? 'done'
        : 'pending'
  if (index < currentIndex.value) return 'done'
  if (index === currentIndex.value)
    return ['failed', 'cancelled'].includes(props.status) ? 'stopped' : 'active'
  return 'pending'
}
</script>

<template>
  <ol class="timeline" aria-label="Steps">
    <li v-for="(stage, index) in STAGES" :key="stage.key" :class="['timeline-step', state(index)]" :title="stage.label">
      <span class="timeline-dot"><i :class="state(index) === 'done' ? 'bi bi-check-lg' : 'bi ' + stage.icon"></i></span>
      <span class="timeline-label">{{ stage.short }}</span>
      <small v-if="stageSeconds[stage.key]" class="timeline-time">{{ durationLabel(stageSeconds[stage.key]) }}</small>
    </li>
  </ol>
</template>
