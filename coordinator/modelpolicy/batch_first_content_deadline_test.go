package modelpolicy

import (
	"testing"
	"time"
)

// TestBatchFirstContentDeadline pins the batch-lane clock: the 120s base plus
// the same per-prompt-token slope the online clock uses, and — unlike the
// online clock — never tightened by an exact-model upstream policy.
func TestBatchFirstContentDeadline(t *testing.T) {
	if BatchFirstContentBase != 120*time.Second {
		t.Fatalf("BatchFirstContentBase=%v, want 120s", BatchFirstContentBase)
	}
	if got := BatchFirstContentDeadline("some-model", 0); got != BatchFirstContentBase {
		t.Fatalf("deadline(prompt=0)=%v, want %v", got, BatchFirstContentBase)
	}
	if got, want := BatchFirstContentDeadline("some-model", 500),
		BatchFirstContentBase+500*time.Millisecond; got != want {
		t.Fatalf("deadline(prompt=500)=%v, want %v", got, want)
	}

	// The exact-model table tightens the ONLINE clock for this build; the batch
	// clock must be unmoved by it.
	online := CoordinatorFirstContentDeadline(Qwen3VL30BA3BInstructModelID, 0, StandardUpstreamFirstContentBase)
	if online >= BatchFirstContentBase {
		t.Fatalf("fixture: online deadline %v is not tightened below the batch base", online)
	}
	if got := BatchFirstContentDeadline(Qwen3VL30BA3BInstructModelID, 0); got != BatchFirstContentBase {
		t.Fatalf("exact-model policy tightened the batch clock: %v, want %v", got, BatchFirstContentBase)
	}
}
