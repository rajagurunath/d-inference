package api

import (
	"testing"

	"github.com/eigeninference/d-inference/coordinator/registry"
)

// TestBatchLaneNeverAcquiresHedge: the governor is bypassed entirely on the
// batch lane, and nothing is acquired (so nothing has to be released). The
// online control proves the same fixture WOULD have acquired a hedge, so the
// lane is what changed the answer.
func TestBatchLaneNeverAcquiresHedge(t *testing.T) {
	srv, _ := testServer(t)
	if srv.hedgeGov == nil {
		t.Fatal("fixture: server has no hedge governor")
	}

	batch := &dispatchState{
		s:                srv,
		model:            "hedge-lane-model",
		lane:             registry.LaneBatch,
		excludeProviders: map[string]struct{}{},
	}
	verdict, acquired := batch.tryAcquireBackupHedge("primary")
	if acquired {
		t.Fatal("batch lane acquired a hedge budget slot")
	}
	if verdict != hedgeSuppressBatchLane {
		t.Fatalf("verdict=%v, want hedgeSuppressBatchLane", verdict)
	}
	if n := srv.hedgeGov.activeHedgeCount(); n != 0 {
		t.Fatalf("active hedges=%d after a batch attempt, want 0", n)
	}

	online := &dispatchState{
		s:                srv,
		model:            "hedge-lane-model",
		lane:             registry.LaneOnline,
		excludeProviders: map[string]struct{}{},
	}
	if _, acquired := online.tryAcquireBackupHedge("primary"); !acquired {
		t.Fatal("online control did not acquire a hedge — fixture no longer proves the lane is the cause")
	}
	srv.hedgeGov.noteHedgeResolved()
}

// TestBatchLaneNeverFeedsReputationLatency: a batch attempt that delivers
// content still contributes no responsiveness sample. Its time-to-first-content
// is measured against a 120s contract on a slot picked for headroom, so it
// describes the batch lane rather than the provider.
func TestBatchLaneNeverFeedsReputationLatency(t *testing.T) {
	online := &registry.PendingRequest{Timing: &registry.RequestTiming{}}
	if !shouldRecordReputationLatency(online, "chunk") {
		t.Fatal("online attempt with content recorded no latency sample")
	}

	batch := &registry.PendingRequest{
		Timing: &registry.RequestTiming{},
		Traits: registry.RequestTraits{Lane: registry.LaneBatch},
	}
	if shouldRecordReputationLatency(batch, "chunk") {
		t.Fatal("batch attempt fed the provider responsiveness EWMA")
	}
}
