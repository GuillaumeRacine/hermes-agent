/**
 * GFM table support (item 7). The native `<code filetype="markdown">` colorizes
 * pipes but never aligns tables into a grid, so we split assistant text into
 * table blocks (rendered by view/mdTable.tsx as an aligned grid) and everything
 * else (rendered by the native markdown renderable). Pure + unit-testable — no
 * OpenTUI/Solid imports. The column-width + alignment algorithm mirrors
 * free-code's MarkdownTable (stringWidth + padAligned).
 */

export type Align = 'left' | 'center' | 'right'

/** A run of non-table markdown, rendered by the native renderable. */
export interface MdSegment {
  readonly kind: 'md'
  readonly text: string
}
/** A parsed GFM table, rendered as an aligned grid. */
export interface TableSegment {
  readonly kind: 'table'
  readonly header: readonly string[]
  readonly rows: readonly (readonly string[])[]
  readonly align: readonly Align[]
}
export type Segment = MdSegment | TableSegment

/** Display width of a string (monospace cells; ASCII-accurate, good enough for CJK). */
export function cellWidth(s: string): number {
  return [...s].length
}

/** Pad `content` (which occupies `width` columns) to `target` columns per `align`. */
export function padAligned(content: string, target: number, align: Align): string {
  const pad = Math.max(0, target - cellWidth(content))
  if (align === 'right') return ' '.repeat(pad) + content
  if (align === 'center') {
    const left = Math.floor(pad / 2)
    return ' '.repeat(left) + content + ' '.repeat(pad - left)
  }
  return content + ' '.repeat(pad)
}

/** Split a GFM row `| a | b |` into trimmed cells (drop the empty edge cells). */
function splitRow(line: string): string[] {
  let s = line.trim()
  if (s.startsWith('|')) s = s.slice(1)
  if (s.endsWith('|')) s = s.slice(0, -1)
  // split on unescaped pipes
  return s.split(/(?<!\\)\|/).map(c => c.trim().replace(/\\\|/g, '|'))
}

/** Is `line` a GFM separator row (e.g. `| --- | :--: | ---: |`)? Returns the per-column align, or null. */
function parseSeparator(line: string): Align[] | null {
  const s = line.trim()
  if (!/^\|?[\s:|-]+\|?$/.test(s) || !s.includes('-')) return null
  const cells = splitRow(line)
  if (cells.length === 0) return null
  const aligns: Align[] = []
  for (const c of cells) {
    const t = c.trim()
    if (!/^:?-+:?$/.test(t)) return null // every cell must be a dash spec
    const left = t.startsWith(':')
    const right = t.endsWith(':')
    aligns.push(left && right ? 'center' : right ? 'right' : 'left')
  }
  return aligns
}

const hasPipe = (line: string) => line.includes('|')

/**
 * Split markdown `text` into ordered md/table segments. A table is a row with a
 * pipe immediately followed by a separator row; data rows continue while they
 * contain a pipe. Incomplete tables (no separator yet — e.g. mid-stream) stay in
 * the md segment and finalize once the separator arrives.
 */
export function segmentMarkdown(text: string): Segment[] {
  const lines = (text ?? '').split('\n')
  const segments: Segment[] = []
  let buf: string[] = []
  const flush = () => {
    if (buf.length) segments.push({ kind: 'md', text: buf.join('\n') })
    buf = []
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!
    const next = lines[i + 1]
    const align = next !== undefined && hasPipe(line) ? parseSeparator(next) : null
    if (align) {
      const header = splitRow(line)
      const cols = Math.max(header.length, align.length)
      const rows: string[][] = []
      let j = i + 2
      for (; j < lines.length; j++) {
        const r = lines[j]!
        if (!hasPipe(r) || r.trim() === '') break
        rows.push(splitRow(r))
      }
      flush()
      segments.push({
        kind: 'table',
        header,
        rows,
        align: Array.from({ length: cols }, (_, k) => align[k] ?? 'left')
      })
      i = j - 1
      continue
    }
    buf.push(line)
  }
  flush()
  return segments
}

/** Column widths for a table, each capped so the whole grid fits `maxWidth`. */
export function tableColumnWidths(seg: TableSegment, maxWidth: number): number[] {
  const cols = seg.align.length
  const natural: number[] = []
  for (let c = 0; c < cols; c++) {
    let w = cellWidth(seg.header[c] ?? '')
    for (const row of seg.rows) w = Math.max(w, cellWidth(row[c] ?? ''))
    natural[c] = Math.max(3, w)
  }
  // budget: each col adds `width + 3` ("│ " + " ") of chrome; shrink proportionally if over.
  const chrome = cols * 3 + 1
  const total = natural.reduce((a, b) => a + b, 0) + chrome
  if (total <= maxWidth) return natural
  const avail = Math.max(cols * 3, maxWidth - chrome)
  const scale = avail / natural.reduce((a, b) => a + b, 0)
  return natural.map(w => Math.max(3, Math.floor(w * scale)))
}
