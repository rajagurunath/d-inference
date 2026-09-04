package registry

import (
	"testing"
)

// TestBatchLaneSkipsTTFTCalibrationJoin: a batch attempt must never join the
// TTFT calibrator. The calibrator learns actual/predicted for the live
// first-content ceiling; a batch attempt runs on a 120s contract and would
// teach it about traffic that ceiling does not govern.
func TestBatchLaneSkipsTTFTCalibrationJoin(t *testing.T) {
	for _, tc := range []struct {
		name         string
		lane         Lane
		wantObserved bool
	}{
		{"online lane joins the calibrator", LaneOnline, true},
		{"batch lane does not", LaneBatch, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ResetTTFTCalibration()
			t.Cleanup(ResetTTFTCalibration)

			reg := New(testLogger())
			model := "calibration-lane-model"
			laneTestProvider(t, reg, "warm", model, 40, 0, 0)

			pr := batchLaneRequest("calib-"+string(tc.lane)+"x", tc.lane)
			p, decision := reg.ReserveProviderEx(model, pr)
			if p == nil {
				t.Fatalf("reservation failed: %+v", decision)
			}
			if _, ok := RecordTTFTObservation(pr.RequestID, pr.Attempt, 5000); ok != tc.wantObserved {
				t.Fatalf("RecordTTFTObservation joined=%v, want %v", ok, tc.wantObserved)
			}
		})
	}
}

// TestBatchLaneSkipsTTFTShadow: the Phase-0 admission shadow measures whether an
// ONLINE request would have missed its upstream SLA. A batch attempt must leave
// it untouched even when its own (long) latency would look like a shed.
func TestBatchLaneSkipsTTFTShadow(t *testing.T) {
	prevMode := TTFTAdmissionModeValue()
	SetTTFTAdmissionMode(TTFTAdmissionShadow)
	t.Cleanup(func() { SetTTFTAdmissionMode(prevMode) })

	reg := New(testLogger())
	model := "shadow-lane-model"
	// Two boxes so the shadow has an idle alternative to report on, and the
	// winner carries occupancy (the branch that walks the pool).
	laneTestProvider(t, reg, "busy", model, 40, 1, 0)
	laneTestProvider(t, reg, "idle", model, 40, 0, 0)

	online := batchLaneRequest("shadow-online", LaneOnline)
	if p, decision := reg.ReserveProviderEx(model, online); p == nil {
		t.Fatalf("online reservation failed: %+v", decision)
	} else if !decision.ShadowEvaluated {
		t.Fatal("shadow was not evaluated for an online request — fixture no longer exercises it")
	}

	batch := batchLaneRequest("shadow-batch", LaneBatch)
	p, decision := reg.ReserveProviderEx(model, batch)
	if p == nil {
		t.Fatalf("batch reservation failed: %+v", decision)
	}
	if decision.ShadowEvaluated {
		t.Fatalf("TTFT shadow evaluated a batch attempt: %+v", decision)
	}
}

// TestBatchLaneIsNeverQueued: the coordinator wait queue is structurally closed
// to the batch lane. A batch item with no headroom is re-offered on a later
// dispatcher tick, never parked against online traffic for the queue's wait.
func TestBatchLaneIsNeverQueued(t *testing.T) {
	q := NewRequestQueue(8, 0)

	batch := &QueuedRequest{
		RequestID:  "q-batch",
		Model:      "queue-lane-model",
		Pending:    batchLaneRequest("q-batch", LaneBatch),
		ResponseCh: make(chan *Provider, 1),
	}
	if err := q.Enqueue(batch); err != ErrBatchLaneNotQueueable {
		t.Fatalf("Enqueue(batch)=%v, want ErrBatchLaneNotQueueable", err)
	}
	if depth := q.QueueSize("queue-lane-model"); depth != 0 {
		t.Fatalf("queue depth=%d after a refused batch enqueue, want 0", depth)
	}

	online := &QueuedRequest{
		RequestID:  "q-online",
		Model:      "queue-lane-model",
		Pending:    batchLaneRequest("q-online", LaneOnline),
		ResponseCh: make(chan *Provider, 1),
	}
	if err := q.Enqueue(online); err != nil {
		t.Fatalf("Enqueue(online)=%v, want nil", err)
	}
	if depth := q.QueueSize("queue-lane-model"); depth != 1 {
		t.Fatalf("queue depth=%d after an online enqueue, want 1", depth)
	}
}
