<script setup>
// The Dubloom mark: a bold D with a sound wave cut through it. Same geometry as /favicon.svg.
defineProps({ size: { type: Number, default: 30 }, plain: { type: Boolean, default: false } })
const bars = [
  [27, 25, 39],
  [34, 19, 45],
  [41, 24, 40],
] // x, top, bottom: the wave punched out of the D
</script>

<template>
  <svg :width="size" :height="size" viewBox="0 0 64 64" aria-hidden="true" class="logo" :class="{ plain }">
    <defs>
      <linearGradient id="dubloom-g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#ff4d6d" /><stop offset=".6" stop-color="#ff8a3d" /><stop offset="1" stop-color="#ffb35c" />
      </linearGradient>
    </defs>
    <rect v-if="!plain" width="64" height="64" rx="18" fill="url(#dubloom-g)" />
    <path d="M15 12h17.5a20 20 0 0 1 0 40H15z" :fill="plain ? 'url(#dubloom-g)' : '#fff'" />
    <g :stroke="plain ? 'var(--panel, #fff)' : 'url(#dubloom-g)'" stroke-width="4.6" stroke-linecap="round">
      <line v-for="([x, top, bottom], i) in bars" :key="i" :x1="x" :y1="top" :x2="x" :y2="bottom" />
    </g>
  </svg>
</template>
