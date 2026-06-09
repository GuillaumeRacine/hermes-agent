/**
 * GFM table segmenter test (item 7). segmentMarkdown splits prose from tables;
 * padAligned/tableColumnWidths drive the aligned grid render.
 */
import { describe, expect, test } from 'bun:test'

import { padAligned, segmentMarkdown, tableColumnWidths, type TableSegment } from '../logic/mdTable.ts'

describe('segmentMarkdown', () => {
  test('splits prose / table / prose and parses header, rows, alignment', () => {
    const segs = segmentMarkdown(
      ['Here is a table:', '', '| Name | Age | City |', '|:-----|----:|:----:|', '| Ann | 30 | NYC |', '| Bo | 5 | LA |', '', 'Done.'].join(
        '\n'
      )
    )
    expect(segs.map(s => s.kind)).toEqual(['md', 'table', 'md'])
    const t = segs[1] as TableSegment
    expect(t.header).toEqual(['Name', 'Age', 'City'])
    expect(t.rows).toEqual([
      ['Ann', '30', 'NYC'],
      ['Bo', '5', 'LA']
    ])
    expect(t.align).toEqual(['left', 'right', 'center'])
    expect((segs[0] as { text: string }).text).toContain('Here is a table:')
    expect((segs[2] as { text: string }).text).toContain('Done.')
  })

  test('plain prose with no table is one md segment', () => {
    const segs = segmentMarkdown('just text\nmore text')
    expect(segs).toHaveLength(1)
    expect(segs[0]!.kind).toBe('md')
  })

  test('a pipe line WITHOUT a separator row is NOT a table (e.g. mid-stream / code)', () => {
    const segs = segmentMarkdown('| not | a | table |\nstill streaming')
    expect(segs.every(s => s.kind === 'md')).toBe(true)
  })

  test('padAligned honors left/right/center', () => {
    expect(padAligned('hi', 6, 'left')).toBe('hi    ')
    expect(padAligned('hi', 6, 'right')).toBe('    hi')
    expect(padAligned('hi', 6, 'center')).toBe('  hi  ')
  })

  test('tableColumnWidths fit within maxWidth (shrink when over budget)', () => {
    const seg: TableSegment = {
      kind: 'table',
      header: ['aaaaaaaaaa', 'bbbbbbbbbb'],
      rows: [['cccccccccc', 'dddddddddd']],
      align: ['left', 'left']
    }
    const widths = tableColumnWidths(seg, 20)
    const chrome = 2 * 3 + 1
    expect(widths.reduce((a, b) => a + b, 0) + chrome).toBeLessThanOrEqual(20)
    expect(Math.min(...widths)).toBeGreaterThanOrEqual(3)
  })
})
