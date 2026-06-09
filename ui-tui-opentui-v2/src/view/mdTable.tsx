/**
 * MdTable — renders a parsed GFM table as an aligned grid (item 7). The native
 * markdown renderable leaves tables as raw pipes; this lays them out with
 * per-column widths, alignment, a header rule, and dim `│` separators. Width
 * is reactive (useTerminalDimensions) so columns shrink to fit on resize.
 */
import { useTerminalDimensions } from '@opentui/solid'
import { createMemo, type JSX } from 'solid-js'

import { padAligned, tableColumnWidths, type TableSegment } from '../logic/mdTable.ts'
import { truncate } from '../logic/toolOutput.ts'
import { useTheme } from './theme.tsx'

const GUTTER = 2

export function MdTable(props: { seg: TableSegment }) {
  const theme = useTheme()
  const dims = useTerminalDimensions()
  const widths = createMemo(() => tableColumnWidths(props.seg, Math.max(24, dims().width - GUTTER - 2)))

  const cellText = (raw: string, i: number) =>
    padAligned(truncate(raw ?? '', widths()[i]!), widths()[i]!, props.seg.align[i] ?? 'left')

  // One row as alternating dim-border / content spans (built as an array so we
  // avoid <For> inside <text>, which doesn't render inline spans reliably).
  const rowSpans = (cells: readonly string[], bold: boolean): JSX.Element[] => {
    const out: JSX.Element[] = []
    const bar = () => <span style={{ fg: theme().color.border }}>│ </span>
    out.push(bar())
    widths().forEach((_, i) => {
      const text = cellText(cells[i] ?? '', i)
      out.push(
        bold ? (
          <span style={{ fg: theme().color.primary }}>
            <b>{text}</b>
          </span>
        ) : (
          <span style={{ fg: theme().color.text }}>{text}</span>
        )
      )
      out.push(<span style={{ fg: theme().color.border }}>{' │ '}</span>)
    })
    return out
  }

  const sepLine = () => `│ ${widths().map(w => '─'.repeat(w)).join('─┼─')} │`

  return (
    <box style={{ flexDirection: 'column', flexShrink: 0 }}>
      <text>{rowSpans(props.seg.header, true)}</text>
      <text>
        <span style={{ fg: theme().color.border }}>{sepLine()}</span>
      </text>
      {props.seg.rows.map(r => (
        <text>{rowSpans(r, false)}</text>
      ))}
    </box>
  )
}
