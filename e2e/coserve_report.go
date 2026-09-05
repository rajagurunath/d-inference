package e2e

// coserve_report.go — the markdown the co-serving benchmark emits.
//
// The benchmark's product is a frozen report under docs/reports/ (docs/AGENTS.md
// §"Reports"): a header, what was measured on what, one metric table, and the
// caveats that bound how far the numbers travel. Rendering it here rather than
// inside the benchmark keeps the test about measurement and makes the table
// shape unit-testable without a provider.

import (
	"fmt"
	"strings"
)

// CoServeMetric is one row of the results table: what was measured, how it is
// defined, and what came out. Definition is prose, not a formula reference —
// a reader of the frozen report has only the report.
type CoServeMetric struct {
	Name       string
	Definition string
	Value      string
}

// CoServeReport is the whole document. Every section is optional except the
// title and the metrics; an empty section renders nothing rather than an empty
// heading.
type CoServeReport struct {
	// Title is the H1, without a leading "# ".
	Title string
	// Intro is the paragraph directly under the title.
	Intro string
	// Setup is a definition list rendered as a two-column table (e.g. host,
	// model, coordinator, provider count).
	Setup []CoServeSetting
	// Method is one bullet per line describing how each phase was run.
	Method []string
	// Metrics is the results table.
	Metrics []CoServeMetric
	// Gates is one bullet per pass/fail gate, already resolved.
	Gates []string
	// Caveats is one bullet per limitation.
	Caveats []string
}

// CoServeSetting is one row of the setup table.
type CoServeSetting struct {
	Name  string
	Value string
}

// MetricsTable renders just the results table: metric, definition, value.
func (r CoServeReport) MetricsTable() string {
	var b strings.Builder
	b.WriteString("| Metric | Definition | Value |\n|---|---|---|\n")
	for _, m := range r.Metrics {
		b.WriteString(fmt.Sprintf("| %s | %s | %s |\n",
			escapeCell(m.Name), escapeCell(m.Definition), escapeCell(m.Value)))
	}
	return b.String()
}

// SetupTable renders the setup rows as a two-column table.
func (r CoServeReport) SetupTable() string {
	if len(r.Setup) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("| Setting | Value |\n|---|---|\n")
	for _, s := range r.Setup {
		b.WriteString(fmt.Sprintf("| %s | %s |\n", escapeCell(s.Name), escapeCell(s.Value)))
	}
	return b.String()
}

// Render assembles the full markdown document.
func (r CoServeReport) Render() string {
	var b strings.Builder
	b.WriteString("# " + r.Title + "\n\n")
	if r.Intro != "" {
		b.WriteString(r.Intro + "\n\n")
	}
	if table := r.SetupTable(); table != "" {
		b.WriteString("## Setup\n\n")
		b.WriteString(table)
		b.WriteString("\n")
	}
	if len(r.Method) > 0 {
		b.WriteString("## Method\n\n")
		writeBullets(&b, r.Method)
		b.WriteString("\n")
	}
	b.WriteString("## Results\n\n")
	b.WriteString(r.MetricsTable())
	b.WriteString("\n")
	if len(r.Gates) > 0 {
		b.WriteString("## Gates\n\n")
		writeBullets(&b, r.Gates)
		b.WriteString("\n")
	}
	if len(r.Caveats) > 0 {
		b.WriteString("## Caveats\n\n")
		writeBullets(&b, r.Caveats)
		b.WriteString("\n")
	}
	return b.String()
}

func writeBullets(b *strings.Builder, lines []string) {
	for _, line := range lines {
		b.WriteString("- " + line + "\n")
	}
}

// escapeCell keeps a value containing a pipe from splitting its table cell.
func escapeCell(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "\n", " "), "|", `\|`)
}
