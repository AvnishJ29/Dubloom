# Voice Library

Forty-five first-party voices across twenty-two Indic languages: the full Rasa
studio roster (44 voices, one female and one male per language, plus a second
Hindi female) and **Suhani**, the Aivanta expressive customer-care voice.
**Every voice speaks every supported language** — conditioning is fully
cross-lingual — but each voice has a native language where accent and diction
are strongest; the recommendations below follow that. Voices are selected by
passing `speaker_id` (the name below); delivery is controlled independently
with a style tag (see *Styles*, bottom).

Each card lists measured register (median pitch), expressiveness (pitch
variability), and speaking rate — plan script length against the rate (library
median ≈ 12 chars/sec). Benchmark-covered voices are measured from model
generations at the production config (ckpt-8944, t 0.6 · top_p 0.9 · rp 1.2);
the remaining voices are measured from their studio source recordings.

## Recommended voices by language

| Language | Female | Male |
|---|---|---|
| Assamese (as) | Prastuti | Ankur |
| Bengali (bn) | Ishita | Sourav |
| Bodo (brx) | Gwrbw | Sansuma |
| Dogri (doi) | Preeti | Sham |
| Gujarati (gu) | Dhara | Parth |
| Hindi (hi) | Kavya · Suhani | Amit |
| Kannada (kn) | Deepika | Adarsh |
| Kashmiri (ks) | Zoon | Ishfaq |
| Konkani (kok) | Anjali | Sandeep |
| Maithili (mai) | Vaidehi | Madhukar |
| Malayalam (ml) | Lakshmi | Kiran |
| Manipuri (mni) | Thoibi | Chaoba |
| Marathi (mr) | Anagha | Chinmay |
| Nepali (ne) | Srijana | Sagar |
| Odia (or) | Itishree | Akash |
| Punjabi (pa) | Kaur | Manpreet |
| Sanskrit (sa) | Bharati | Aryaman |
| Santali (sat) | Phulmani | Sibu |
| Sindhi (sd) | Moomal | Rano |
| Tamil (ta) | Anitha | Arun |
| Telugu (te) | Sravani | Vamsi |
| Urdu (ur) | Saba | Zaid |

---

## Assamese

**Prastuti** — Female · ~216 Hz · 12.5 chars/sec. Warm mid register with
animated inflection; friendly and engaging. *Best for:* e-learning,
storytelling, promos.

**Ankur** — Male · ~108 Hz · 11.9 chars/sec. The deepest voice in the library,
with strikingly expressive delivery — dramatic bass presence. *Best for:*
dramatic audiobooks, trailers, documentaries.

## Bengali

**Ishita** — Female · ~266 Hz · 11.6 chars/sec. Bright, youthful upper
register with lively intonation. *Best for:* children's stories, audiobooks,
engaging e-learning.

**Sourav** — Male · ~142 Hz · 13.2 chars/sec. Rich low register, confident and
naturally varied — a dependable anchor sound. *Best for:* TV-style news,
documentaries, corporate narration. Data note: uniquely deep long-form
narration reserves (4 h of heritage-reading recordings beyond the standard mix).

## Bodo

**Gwrbw** — Female · ~265 Hz · 12.4 chars/sec. Bright register with a notably
even, composed contour. *Best for:* news bulletins, announcements, e-learning.

**Sansuma** — Male · ~150 Hz · 12.1 chars/sec. Low register with animated
inflection; earnest storyteller energy. *Best for:* audiobooks, human-interest
narration.

## Dogri

**Preeti** — Female · ~274 Hz · 13.0 chars/sec. Bright and brisk with balanced
expressiveness; crisp and modern. *Best for:* TV-style news, promos,
explainers.

**Sham** — Male · ~151 Hz · 12.0 chars/sec. Low register, warmly expressive.
*Best for:* audiobooks, AIR-style news, narration.

## Gujarati

**Dhara** — Female · ~273 Hz · 10.7 chars/sec. Among the brightest voices,
with the most unhurried delivery — patient and deliberate. *Best for:*
children's stories, step-by-step tutorials, IVR.

**Parth** — Male · ~164 Hz · 12.7 chars/sec. Mid tenor, animated and
persuasive. *Best for:* advertisements, promos, energetic e-learning.

## Hindi

**Kavya** — Female · ~215 Hz · 12.9 chars/sec. Balanced low-mid register,
composed and professional. *Best for:* TV-style news, corporate narration,
e-learning.

**Suhani** — Female · ~250 Hz · 13.3 chars/sec. The Aivanta expressive
customer-care voice: bright, brisk, unfailingly courteous, trained on
long-form service dialogues. *Best for:* customer care and IVR, social-media
and e-commerce content.

**Amit** — Male · ~152 Hz · 12.6 chars/sec. Mid register, balanced
expressiveness — the all-rounder. *Best for:* e-learning and lectures,
AIR-style news, explainers.

## Kannada

**Deepika** — Female · ~209 Hz · 10.9 chars/sec. Warm alto with a calm, even
contour and relaxed pace — the most soothing voice in the library. *Best
for:* audiobooks, wellness content, children's stories.

**Adarsh** — Male · ~144 Hz · 12.0 chars/sec. Low register with the most even
male contour; grounded and trustworthy. *Best for:* audiobooks, long-form
narration, documentation.

## Kashmiri

**Zoon** — Female · ~284 Hz · 11.9 chars/sec. High, bright register with
strongly animated delivery. Lightest training footprint in the roster.
*Best for:* storytelling, expressive content. Data note: no news or
audiobook tag data at all; emotions ~0.5 h each — conversational-leaning.

**Ishfaq** — Male · ~112 Hz · 14.0 chars/sec. Deep and fast with intense,
expressive drive. *Best for:* dramatic narration, sports-style energy, promos.

## Konkani

**Anjali** — Female · ~239 Hz · 12.4 chars/sec. Clear mid-bright register,
composed and cordial. *Best for:* news, e-learning, announcements.

**Sandeep** — Male · ~135 Hz · 13.7 chars/sec. Deep, quick and animated;
persuasive presence. *Best for:* advertisements, TV-style news, promos.

## Maithili

**Vaidehi** — Female · ~256 Hz · 11.9 chars/sec. Bright register with
expressive warmth. *Best for:* storytelling, audiobooks, cultural programming.
Data note: negative-emotion tags are thin (0.3–0.5 h) — happy/sad are solid.

**Madhukar** — Male · ~231 Hz · 9.9 chars/sec. Distinctive high tenor with
highly expressive, unhurried delivery — a traditional reciter's character.
*Best for:* poetry and recitation, folk storytelling, devotional content.

## Malayalam

**Lakshmi** — Female · ~237 Hz · 13.2 chars/sec. Clear mid register with the
steadiest female contour; newsroom-neutral. *Best for:* AIR-style bulletins,
announcements, factual narration.

**Kiran** — Male · ~131 Hz · 14.0 chars/sec. Deep register at the fastest male
pace; energetic authority. *Best for:* TV-style news, live-feel segments,
promos.

## Manipuri

**Thoibi** — Female · ~211 Hz · 9.3 chars/sec. Warm alto with the gentlest
pace in the library and a serene, even contour. *Best for:* audiobooks,
meditation and wellness, children's bedtime stories.

**Chaoba** — Male · ~115 Hz · 10.4 chars/sec. Deep, deliberate and expressive;
quiet gravitas. *Best for:* documentaries, solemn narration, audiobooks.

## Marathi

**Anagha** — Female · ~261 Hz · 11.7 chars/sec. Bright register, gently
varied, polished — classic broadcast host. *Best for:* AIR-style news, event
hosting, e-learning.

**Chinmay** — Male · ~173 Hz · 12.6 chars/sec. Bright tenor, warm and
articulate — approachable teacher energy. *Best for:* e-learning, children's
content, friendly explainers.

## Nepali

**Srijana** — Female · ~292 Hz · 11.1 chars/sec. The highest register in the
library, delivered with composed steadiness — light and clear. *Best for:*
children's stories, announcements, e-learning.

**Sagar** — Male · ~178 Hz · 11.8 chars/sec. The most expressive voice in the
entire library — sweeping, theatrical intonation. *Best for:* dramatic
audiobooks, advertisements, performance narration.

## Odia

**Itishree** — Female · ~263 Hz · 11.5 chars/sec. Bright, open register with
animated inflection; friendly storyteller. *Best for:* children's stories,
audiobooks, community announcements.

**Akash** — Male · ~151 Hz · 11.8 chars/sec. Low-mid register, sincere and
engaging. *Best for:* audiobooks, AIR-style news, human-interest stories.

## Punjabi

**Kaur** — Female · ~254 Hz · 11.9 chars/sec. Bright register with a composed,
even contour; assured and clear. *Best for:* news, e-learning, announcements.

**Manpreet** — Male · ~184 Hz · 13.3 chars/sec. Upper-mid tenor, brisk and
steady; bright professional energy. *Best for:* TV-style news, promos,
explainers.

## Sanskrit

**Bharati** — Female · ~284 Hz · 10.5 chars/sec. High, luminous register with
measured, composed delivery — a chant-like clarity. *Best for:* recitation,
devotional and educational content.

**Aryaman** — Male · ~147 Hz · 9.5 chars/sec. Low register at a slow,
weighty, expressive pace — oratorical gravity. *Best for:* shloka recitation,
solemn narration, documentaries.

## Santali

**Phulmani** — Female · ~284 Hz · 12.3 chars/sec. High, bright register with
lively movement. *Best for:* storytelling, community content, e-learning.

**Sibu** — Male · ~166 Hz · 12.4 chars/sec. Mid register, strongly expressive.
*Best for:* storytelling, radio-style segments.

## Sindhi

**Moomal** — Female · ~265 Hz · 12.0 chars/sec. Bright register, balanced
delivery. Lighter training footprint than most voices. *Best for:* general
narration, announcements.

**Rano** — Male · ~157 Hz · 13.2 chars/sec. Low-mid register, quick and
animated. *Best for:* news, promos, energetic narration.

## Tamil

**Anitha** — Female · ~210 Hz · 13.2 chars/sec. Warm alto, even contour,
quick; composed authority. *Best for:* TV-style news, explainers,
professional narration. Data note: the emotion standout of the roster —
1.5–2× everyone's per-emotion training hours (anger/happy ~2.2 h each).

**Arun** — Male · ~121 Hz · 14.5 chars/sec. Very deep, very fast, and
strongly expressive — commanding, dramatic presence. *Best for:*
advertisements, dramatic audiobooks, trailers.

## Telugu

**Sravani** — Female · ~260 Hz · 12.3 chars/sec. Bright register, among the
most expressive female contours; vivid and engaging. *Best for:*
advertisements, children's stories, expressive e-learning.

**Vamsi** — Male · ~143 Hz · 11.8 chars/sec. Rich low register, highly
expressive; warm gravitas. *Best for:* audiobooks, storytelling, emotional
narration.

## Urdu

**Saba** — Female · ~209 Hz · 11.8 chars/sec. Velvet alto, balanced and
elegant; literary polish. *Best for:* audiobooks, poetry and literature,
cultural programming. Data note: essentially no emotion-tag training
(anger 0 h, happy/sad 0.1 h) — do not pair with emotion styles.

**Zaid** — Male · ~127 Hz · 12.8 chars/sec. Deep, refined and expressive;
formal weight. *Best for:* news bulletins, poetry, formal narration.

---

## Emotion & style coverage (measured from the training data)

Speaker discovery over the training manifests (2026-08-26), so recommendations
below are evidence, not vibes:

- **Emotions** (`anger` `disgust` `fear` `happy` `sad` `surprise`): every Rasa
  voice carries ~1 h per emotion — emotion styles work roster-wide.
  Exceptions: **Saba** (none — see card), **Zoon**/**Moomal** (thin, ~0.2–0.5 h),
  **Vaidehi** (negative emotions thin). Standout: **Anitha** (1.5–2× depth).
- **`news`**: all voices except Zoon (~2 h each). Deepest: Phulmani (3.0 h),
  Ankur (2.8), Kavya (2.7), Rano (2.7), Madhukar (2.6).
- **`single person narration audiobook`**: all voices except Zoon (~2 h each).
  Deepest: Anagha and Phulmani (2.7 h).
- **Conversational speech**: all main voices carry ~1.8 k conversation turns
  (untagged — it is coverage, not a steerable tag).
- **Suhani**: `Customer Care` 3.1 h + companion-persona 1.0 h only — no
  emotion or news tags; keep her on service/social content.

## Styles

Any voice combines with any trained style tag (passed as `style`):

`advertisements` · `AIR style news` · `TV style news` ·
`single person narration audiobook` · `children's stories` ·
`educational lecture` · `news` · `Customer Care` ·
emotions (`happy`, `sad`, `anger`, `fear`, `surprise`, `disgust`)

Script-writing guidance: plan length with the voice's chars/sec (library
median ≈ 12), and punctuate phonetically — the model derives pauses and
intonation from punctuation, so place commas wherever a speaker would breathe,
not only where grammar requires.

The styled checkpoints (llama3_yt_styled_*) additionally accept measured
prose captions in the `style` field — register phrase + pace clause + voice
structure + channel, e.g. `"Energetic TV news broadcast speech, at a rapid,
urgent pace, one narrator carrying the whole passage, clean broadcast-studio
sound."` Pace clauses only reach their extremes on voices whose recordings
span that pace; register and pitch-dynamism clauses work across all voices.

## News template (urgent TV bulletin)

Validated on the styled checkpoint (2026-08-26):

- **Caption:** `Energetic TV news broadcast speech, at a rapid, urgent pace,
  one narrator carrying the whole passage, clean broadcast-studio sound.`
  (alt registers: `Urgent television news delivery`, `Forceful broadcast
  news delivery`; alt pace: `at a brisk, driving pace`.)
- **Punctuation:** write the bulletin as segments; end EVERY segment with
  `!` or the script's sentence mark (`।`/`۔`) — these carry the anchor's
  segment pauses. Inside a segment, run on: link clauses with và/और-style
  conjunctions and at most a snatched-breath comma. Removing segment marks
  does NOT speed the read; it only breaks the bulletin rhythm.
- **Length:** chars ≈ 12 × target-seconds for most voices; Parth sustains
  ~14–15 chars/sec on this template, so budget accordingly.
- **Voices:** **Parth** for urgent bulletins (fastest, 14.8 c/s measured on
  this template, cross-lingual), **Amit** for prime-time gravitas (most
  expressive, 7.2 st pitch spread), Kiran for TV energy, Lakshmi for
  measured AIR-style bulletins.
- **Sampling:** expressiveness varies seed-to-seed at t 1.0 — for hero
  clips, generate 2–3 seeds and keep the read with the widest pitch spread.
