package api

import (
	"bytes"
	"log/slog"
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/protocol"
	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/store"
)

// batchProbePlanState builds a dispatchState holding a REAL retained plan with
// alternates left in it, on the given lane. Everything else — the request
// clock, the deadline, the absence of a self-route policy — is what
// maybeProbePlanCandidates needs to reach its probe round.
func batchProbePlanState(t *testing.T, lane registry.Lane) *dispatchState {
	t.Helper()
	const model = "batch-probe-model"
	logger := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelError}))
	reg := registry.New(logger)
	srv := NewServer(reg, store.NewMemory(store.Config{AdminKey: "test-key"}), ServerConfig{}, logger)

	for _, id := range []string{"p-probe-a", "p-probe-b", "p-probe-c"} {
		p := makeRoutableProvider(t, reg, id, model)
		p.Mu().Lock()
		p.BackendCapacity = &protocol.BackendCapacity{
			TotalMemoryGB: 64,
			Slots: []protocol.BackendSlotCapacity{
				{Model: model, State: "running", NumRunning: 0, NumWaiting: 0},
			},
		}
		p.Mu().Unlock()
	}

	pr := &registry.PendingRequest{
		RequestID: "req-probe", Model: model,
		EstimatedPromptTokens: 64, RequestedMaxTokens: 128,
		ChunkCh:    make(chan registry.ProviderChunk, 1),
		CompleteCh: make(chan protocol.UsageInfo, 1),
		ErrorCh:    make(chan protocol.InferenceErrorMessage, 1),
		Traits:     registry.RequestTraits{Lane: lane},
	}
	provider, _, plan := reg.ReserveProviderWithPlan(model, pr)
	if provider == nil {
		t.Fatal("fixture: nothing reserved, so there is no plan to probe")
	}
	t.Cleanup(func() { srv.releaseUnsentDispatch(provider, pr) })
	if plan == nil || plan.Len() == 0 {
		t.Fatal("fixture: the reservation retained no alternates, so a probe round would be a no-op anyway")
	}

	return &dispatchState{
		s: srv, model: model, publicModel: model,
		lane:                  lane,
		plan:                  plan,
		estimatedPromptTokens: 64,
		requestedMaxTokens:    128,
		deadline:              30 * time.Second,
		speculativeAt:         5 * time.Second,
		timing:                &registry.RequestTiming{ReceivedAt: time.Now()},
	}
}

// TestBatchLaneNeverProbesPlanCandidates is the B5 regression. The one
// capacity-probe round exists to refine the HEDGE instant for the retained
// alternates — and a batch attempt never hedges (skipSpeculativeBackup) and
// never queues, so nothing can consume the answer. Fanning capacity_probe
// frames anyway spends a round trip on every alternate, charged to the online
// fleet's slots on behalf of traffic that cannot use them.
//
// probesLaunched is set on the statement immediately before
// registry.ProbePlanCandidates and nowhere else, so "still false" is exactly
// "no capacity_probe frame was ever built for this request".
func TestBatchLaneNeverProbesPlanCandidates(t *testing.T) {
	batch := batchProbePlanState(t, registry.LaneBatch)
	batch.maybeProbePlanCandidates()
	if batch.probesLaunched {
		t.Error("a batch dispatch fanned a capacity-probe round its own hedge suppression can never use")
	}
	if batch.hedgeAdvanceCh != nil {
		t.Error("a batch dispatch armed the hedge-advance channel")
	}

	// Online control on the same three-provider fixture: the probe round IS
	// launched, so the assertion above is the lane and not an empty plan.
	online := batchProbePlanState(t, registry.LaneOnline)
	online.maybeProbePlanCandidates()
	if !online.probesLaunched {
		t.Fatal("online control launched no probe round — the fixture no longer proves the lane is the cause")
	}
}
