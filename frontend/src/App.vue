<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { post, request, subscribe } from './lib/api'
import { date, friendlyStage } from './lib/format'
import { generations } from './lib/history'
import CreatePanel from './components/CreatePanel.vue'
import Logo from './components/Logo.vue'
import JobView from './components/JobView.vue'

const config = ref(null)
const health = ref(null)
const theme = ref(
  localStorage.getItem('dubloom-theme') ||
    (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
)
const projects = ref([])
const selectedId = ref('')
const generation = ref(0) // the generation playing on the job page: 0 = the latest
const job = ref(null)
const prefill = ref(null)
const toasts = ref([])
const needsLogin = ref(false)
const password = ref('')
const loggingIn = ref(false)
const query = ref('')
const sidebarOpen = ref(false)
const offline = ref(false)
const loaded = ref(false) // the list has answered at least once
let unsubscribe = null
let listTimer = null
let disposed = false
let selectionVersion = 0

const ready = computed(() => !!(health.value?.bodhan_configured && health.value?.ffmpeg_found))
const activeCount = computed(
  () => projects.value.filter((item) => ['queued', 'running'].includes(item.status)).length,
)
const filtered = computed(() => {
  const needle = query.value.trim().toLowerCase()
  if (!needle) return projects.value
  return projects.value.filter((item) =>
    [
      item.title,
      item.source_name,
      item.source_url,
      config.value?.languages?.[item.language],
      item.language,
    ]
      .filter(Boolean)
      .some((text) => String(text).toLowerCase().includes(needle)),
  )
})
const languageName = (code) => config.value?.languages?.[code] || code

watch(
  theme,
  (value) => {
    document.documentElement.dataset.theme = value
    localStorage.setItem('dubloom-theme', value)
    document
      .querySelector('meta[name=theme-color]')
      ?.setAttribute('content', value === 'dark' ? '#0c0e16' : '#f5f6fa')
  },
  { immediate: true },
)

function notify(message, kind = 'error') {
  const id = Date.now() + Math.random()
  toasts.value.push({ id, message, kind })
  setTimeout(() => dismiss(id), kind === 'error' ? 8000 : 4000)
}
function dismiss(id) {
  toasts.value = toasts.value.filter((toast) => toast.id !== id)
}
function handleError(error) {
  if (error?.status === 401) {
    needsLogin.value = true
    return
  }
  if (error?.status === 0 || error instanceof TypeError) {
    offline.value = true
    return
  }
  notify(error?.message || String(error))
}
function statusLabel(item) {
  if (item.status === 'queued') return 'Waiting to start'
  if (item.status === 'running')
    return `${friendlyStage(item)} · ${Math.round(item.progress || 0)}%`
  if (item.status === 'completed') return 'Ready'
  if (item.status === 'cancelled') return 'Stopped'
  return 'Needs attention'
}

function chime() {
  // two soft notes, only if the browser lets a page make a sound
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)()
    ;[
      [659.25, 0],
      [880, 0.14],
    ].forEach(([freq, at]) => {
      const osc = ctx.createOscillator(),
        gain = ctx.createGain()
      osc.type = 'sine'
      osc.frequency.value = freq
      gain.gain.setValueAtTime(0.0001, ctx.currentTime + at)
      gain.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + at + 0.02)
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + at + 0.35)
      osc.connect(gain).connect(ctx.destination)
      osc.start(ctx.currentTime + at)
      osc.stop(ctx.currentTime + at + 0.4)
    })
    setTimeout(() => ctx.close(), 1200)
  } catch (error) {
    /* no sound: fine */
  }
}
async function loadList() {
  const before = new Map(projects.value.map((item) => [item.id, item.status]))
  const slow = setTimeout(() => {
    if (!loaded.value) offline.value = true
  }, 15000) // a list that never arrives is a reachability problem, not an empty studio
  try {
    projects.value = await request('/jobs')
    offline.value = false
    loaded.value = true
  } catch (error) {
    handleError(error)
    return
  } finally {
    clearTimeout(slow)
  }
  for (const item of projects.value) {
    // a dub that just finished: say so, wherever the user is in the studio
    if (item.status === 'completed' && ['queued', 'running'].includes(before.get(item.id))) {
      notify(`"${item.title}" is ready to watch`, 'success')
      chime()
    }
  }
}
function scheduleList() {
  clearTimeout(listTimer)
  if (disposed) return
  listTimer = setTimeout(
    async () => {
      await loadList()
      scheduleList()
    },
    activeCount.value ? 5000 : 20000,
  )
}
function mergeSummary(data) {
  const summary = { ...data, segment_count: data.segments?.length || 0 }
  delete summary.segments
  const index = projects.value.findIndex((item) => item.id === data.id)
  if (index >= 0) projects.value.splice(index, 1, { ...projects.value[index], ...summary })
  else projects.value.unshift(summary)
}
function watchJob(id) {
  unsubscribe?.()
  unsubscribe = null
  if (!id) return
  unsubscribe = subscribe(id, {
    onData: (data) => {
      if (selectedId.value === id) {
        job.value = data
        mergeSummary(data)
      }
    },
    onDone: () => loadList(),
    onError: (error) => {
      handleError(error)
      if (error?.status === 404 && selectedId.value === id) select('')
    },
  })
}
async function pickVersion(id, number) {
  if (id !== selectedId.value) await select(id)
  generation.value = number
}
async function select(id, { replace = false } = {}) {
  if (id !== selectedId.value) generation.value = 0
  document.querySelectorAll('video, audio').forEach((media) => media.pause())
  const version = ++selectionVersion
  unsubscribe?.()
  unsubscribe = null
  selectedId.value = id
  job.value = null
  sidebarOpen.value = false
  const target = id ? `?job=${id}` : window.location.pathname
  history[replace ? 'replaceState' : 'pushState']({ job: id }, '', target)
  if (id) {
    try {
      const data = await request('/jobs/' + id)
      if (version !== selectionVersion) return
      job.value = data
    } catch (error) {
      if (version !== selectionVersion) return
      handleError(error)
      if (error?.status === 404) {
        selectedId.value = ''
        history.replaceState({}, '', window.location.pathname)
      }
    }
  } else {
    window.scrollTo({ top: 0 })
  }
  if (version === selectionVersion) watchJob(selectedId.value)
}
function reuse(source) {
  prefill.value = { ...source, title: '', title_shown: source.title } // the new dub gets its own name; the card shows which video
  select('')
}
async function created(id) {
  prefill.value = null
  notify('Your dub has started.', 'success')
  await select(id)
  loadList()
  scheduleList()
}
async function refreshSelected() {
  await loadList()
  const id = selectedId.value
  if (id) {
    try {
      const data = await request('/jobs/' + id)
      if (selectedId.value === id) job.value = data
    } catch (error) {
      handleError(error)
    }
    if (selectedId.value === id) watchJob(id)
  }
  scheduleList()
}
async function deleted() {
  notify('Dub deleted.', 'success')
  await select('')
  loadList()
}
async function clearAll() {
  if (!window.confirm('Delete every dub in the library, including the finished videos?')) return
  try {
    await request('/jobs', { method: 'DELETE' })
    await select('')
    await loadList()
    notify('Library cleared.', 'success')
  } catch (error) {
    handleError(error)
  }
}
async function login() {
  loggingIn.value = true
  try {
    await post('/login', { password: password.value })
    needsLogin.value = false
    password.value = ''
    await init()
  } catch (error) {
    notify(error.message)
  } finally {
    loggingIn.value = false
  }
}
async function init() {
  try {
    const [configuration, status] = await Promise.all([request('/config'), request('/health')])
    config.value = configuration
    health.value = status
    offline.value = false
  } catch (error) {
    handleError(error)
    offline.value = true
    setTimeout(init, 5000)
    return
  } // the sidebar says it is retrying: it is
  await loadList()
  const id = new URLSearchParams(window.location.search).get('job')
  if (id && /^[0-9a-f-]{36}$/i.test(id)) await select(id, { replace: true })
  scheduleList()
}
function onPopState() {
  const id = new URLSearchParams(window.location.search).get('job') || ''
  if (id !== selectedId.value) select(id, { replace: true })
}
onMounted(() => {
  window.addEventListener('popstate', onPopState)
  init()
})
onUnmounted(() => {
  disposed = true
  clearTimeout(listTimer)
  unsubscribe?.()
  window.removeEventListener('popstate', onPopState)
})
</script>

<template>
  <div :class="['shell', { 'sidebar-open': sidebarOpen }]">
    <a class="skip-link" href="#studio-main">Skip to workspace</a>
    <header class="topbar">
      <button class="icon-button menu-toggle" :aria-label="sidebarOpen ? 'Hide your dubs' : 'Show your dubs'" :aria-expanded="sidebarOpen" aria-controls="studio-library" @click="sidebarOpen = !sidebarOpen"><i class="bi bi-list"></i></button>
      <a href="/" class="brand" @click.prevent="select('')">
        <Logo :size="30" class="brand-mark" />
        <span class="brand-name">Dubloom</span>
      </a>
      <nav class="top-right">
        <span v-if="activeCount" class="status-pill busy"><i class="bi bi-arrow-repeat spin"></i>{{ activeCount }} dubbing now</span>
        <span v-if="offline" class="status-pill warn"><i class="status-dot"></i>Can't reach the studio</span>
        <button class="icon-button" :aria-label="theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'" @click="theme = theme === 'light' ? 'dark' : 'light'"><i :class="theme === 'light' ? 'bi bi-moon-stars' : 'bi bi-sun'"></i></button>
      </nav>
    </header>

    <div class="workspace">
      <div class="scrim" @click="sidebarOpen = false"></div>
      <aside id="studio-library" class="library" aria-label="Your dubs" @keydown.esc="sidebarOpen = false">
        <button class="button primary block" @click="select('')"><i class="bi bi-plus-lg"></i> New dub</button>
        <label class="search"><i class="bi bi-search"></i><input v-model="query" type="search" placeholder="Find a dub" aria-label="Find a dub" /></label>
        <div class="library-label">Your dubs <span>{{ projects.length }}</span></div>
        <p v-if="!loaded" class="library-empty">{{ offline ? "Can't reach the studio right now. Your dubs are safe; retrying." : "Loading your dubs…" }}</p>
        <p v-else-if="!projects.length" class="library-empty"><i class="bi bi-camera-reels library-empty-icon" aria-hidden="true"></i>Nothing here yet. Paste a link to make your first dub.</p>
        <p v-else-if="!filtered.length" class="library-empty">No dub matches that.</p>
        <nav class="project-list">
          <template v-for="item in filtered" :key="item.id">
          <button :class="['project', 'status-' + item.status, { selected: selectedId === item.id }]" @click="select(item.id)">
            <span class="thumb"><img v-if="item.poster_url" :src="item.poster_url" alt="" loading="lazy" /><i v-else class="bi bi-film"></i></span>
            <span class="project-copy">
              <strong>{{ item.title || languageName(item.language) + ' dub' }}</strong>
              <small><i class="status-dot"></i>{{ statusLabel(item) }}</small>
              <small class="project-meta">{{ languageName(item.language) }} · {{ date(item.finished_at || item.updated_at || item.created_at) }}</small>
              <small v-if="generations(item).length > 1" class="project-change"><i class="bi bi-layers" aria-hidden="true"></i>{{ generations(item).length }} versions</small>
              <span v-if="['queued', 'running'].includes(item.status)" class="mini-bar"><span :style="{ width: (item.progress || 0) + '%' }"></span></span>
            </span>
          </button>
          <ol v-if="selectedId === item.id && generations(item).length > 1" class="generations" aria-label="Versions of this dub">
            <li v-for="row in generations(item)" :key="row.number">
              <button type="button" :class="{ on: generation === row.number }" @click="pickVersion(item.id, row.number)">
                <strong>{{ row.name }}</strong>
                <small>{{ row.made }}</small>
                <small class="when">{{ date(row.at) }}</small>
              </button>
            </li>
          </ol>
          </template>
        </nav>
        <footer class="library-foot">
          <span><i class="bi bi-collection-play"></i> Saved in your library</span>
          <button v-if="projects.length" class="link-button" @click="clearAll">Clear all</button>
        </footer>
      </aside>

      <main id="studio-main" tabindex="-1">
        <div v-if="offline" class="connection-banner" role="status"><span>Connection lost. Reconnecting to your studio…</span><button class="button small" @click="init">Retry now</button></div>
        <CreatePanel v-if="!selectedId" :config="config" :health="health" :ready="ready" :prefill="prefill" :recent="projects" @created="created" @error="notify($event)" @open="select($event)" />
        <JobView v-else-if="job" :key="job.id" :job="job" :config="config" :version="generation" @version="number => generation = number" @refresh="refreshSelected" @reuse="reuse(job)" @notify="(message, kind) => notify(message, kind || 'error')" @deleted="deleted" />
        <div v-else class="loading"><i class="bi bi-arrow-repeat spin"></i> Opening…</div>
      </main>
    </div>

    <div class="toasts" aria-live="polite">
      <div v-for="toast in toasts" :key="toast.id" :class="['toast', toast.kind]">
        <i :class="'bi ' + (toast.kind === 'success' ? 'bi-check-circle' : 'bi-exclamation-circle')"></i>
        <span>{{ toast.message }}</span>
        <button class="icon-button" aria-label="Dismiss" @click="dismiss(toast.id)"><i class="bi bi-x-lg"></i></button>
      </div>
    </div>

    <div v-if="needsLogin" class="modal-backdrop">
      <form class="modal" @submit.prevent="login">
        <Logo :size="52" class="brand-mark large" />
        <h2>This studio is private</h2>
        <p>Enter the studio password to continue.</p>
        <label class="field"><span>Password</span><input v-model="password" type="password" autocomplete="current-password" autofocus required /></label>
        <button class="button primary block" type="submit" :disabled="loggingIn">{{ loggingIn ? 'Checking…' : 'Continue' }}</button>
      </form>
    </div>
  </div>
</template>
