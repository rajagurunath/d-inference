package testbed

import (
	"strings"
	"testing"
)

func TestCoServeReportMetricsTable(t *testing.T) {
	report := CoServeReport{
		Title: "Batch co-serving benchmark",
		Metrics: []CoServeMetric{
			{Name: "harvest", Definition: "co-serving items/s ÷ offline ceiling", Value: "0.41 (41%)"},
			{Name: "online p99 ratio", Definition: "co-serving p99 ÷ online-only p99", Value: "1.38×"},
		},
	}

	table := report.MetricsTable()
	lines := strings.Split(strings.TrimSuffix(table, "\n"), "\n")
	if len(lines) != 4 {
		t.Fatalf("got %d table lines, want header + separator + 2 rows:\n%s", len(lines), table)
	}
	if lines[0] != "| Metric | Definition | Value |" || lines[1] != "|---|---|---|" {
		t.Fatalf("unexpected table header:\n%s", table)
	}
	if !strings.Contains(lines[2], "harvest") || !strings.Contains(lines[2], "0.41 (41%)") {
		t.Fatalf("first row lost its values: %q", lines[2])
	}
}

func TestCoServeReportEscapesPipes(t *testing.T) {
	report := CoServeReport{Metrics: []CoServeMetric{
		{Name: "a|b", Definition: "one\ntwo", Value: "x|y"},
	}}
	row := strings.Split(strings.TrimSuffix(report.MetricsTable(), "\n"), "\n")[2]
	if strings.Count(row, "|") != 4+2 {
		t.Fatalf("cell pipes were not escaped: %q", row)
	}
	if strings.Contains(row, "\n") {
		t.Fatalf("newline survived into a cell: %q", row)
	}
}

func TestCoServeReportRenderSkipsEmptySections(t *testing.T) {
	report := CoServeReport{
		Title:   "Batch co-serving benchmark",
		Metrics: []CoServeMetric{{Name: "harvest", Definition: "d", Value: "v"}},
	}
	out := report.Render()
	if !strings.HasPrefix(out, "# Batch co-serving benchmark\n") {
		t.Fatalf("missing H1:\n%s", out)
	}
	for _, heading := range []string{"## Setup", "## Method", "## Gates", "## Caveats"} {
		if strings.Contains(out, heading) {
			t.Fatalf("%s rendered with no content:\n%s", heading, out)
		}
	}
	if !strings.Contains(out, "## Results") {
		t.Fatalf("results section missing:\n%s", out)
	}
}

func TestCoServeReportRenderIncludesEverySection(t *testing.T) {
	report := CoServeReport{
		Title:   "T",
		Intro:   "intro line",
		Setup:   []CoServeSetting{{Name: "model", Value: "m/one"}},
		Method:  []string{"phase one", "phase two"},
		Metrics: []CoServeMetric{{Name: "harvest", Definition: "d", Value: "v"}},
		Gates:   []string{"gate one"},
		Caveats: []string{"caveat one"},
	}
	out := report.Render()
	for _, want := range []string{
		"intro line", "## Setup", "| model | m/one |", "## Method", "- phase one",
		"## Results", "## Gates", "- gate one", "## Caveats", "- caveat one",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered report missing %q:\n%s", want, out)
		}
	}
}
