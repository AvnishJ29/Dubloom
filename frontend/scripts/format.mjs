/** Format source scripts while preserving Vue templates and scoped styles byte for byte. */
import { readdir, readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { parse } from '@vue/compiler-sfc'
import prettier from 'prettier'

const root = fileURLToPath(new URL('../', import.meta.url))
const check = process.argv.includes('--check')
const options = {
  quoteProps: 'preserve',
  singleQuote: true,
  semi: false,
  printWidth: 100,
  trailingComma: 'all',
}

async function filesIn(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const nested = await Promise.all(
    entries.map((entry) => {
      const filename = path.join(directory, entry.name)
      return entry.isDirectory() ? filesIn(filename) : [filename]
    }),
  )
  return nested.flat()
}

let changed = 0
const files = [
  ...(await filesIn(path.join(root, 'src'))),
  ...(await filesIn(path.join(root, 'tests'))),
  ...(await filesIn(path.join(root, 'scripts'))),
  path.join(root, 'vite.config.js'),
]

for (const filename of files) {
  if (!/\.(?:vue|m?js|css)$/.test(filename)) continue
  const source = await readFile(filename, 'utf8')
  let formatted
  if (filename.endsWith('.vue')) {
    const { descriptor, errors } = parse(source, { filename })
    if (errors.length) throw new Error(`Cannot parse ${filename}: ${errors.join(', ')}`)
    const blocks = [descriptor.script, descriptor.scriptSetup]
      .filter(Boolean)
      .sort((a, b) => b.loc.start.offset - a.loc.start.offset)
    formatted = source
    for (const block of blocks) {
      const content = '\n' + (await prettier.format(block.content, { ...options, parser: 'babel' }))
      formatted =
        formatted.slice(0, block.loc.start.offset) + content + formatted.slice(block.loc.end.offset)
    }
  } else {
    formatted = await prettier.format(source, { ...options, filepath: filename })
  }
  if (formatted === source) continue
  changed++
  console.log(path.relative(root, filename))
  if (!check) await writeFile(filename, formatted)
}
console.log(`${changed} file(s) ${check ? 'need formatting' : 'formatted'}.`)
if (check && changed) process.exitCode = 1
