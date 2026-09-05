package registry

import (
	"fmt"
	"testing"
)

// laneTestProvider registers a synthetic provider for model and stamps the live
// slot counters the batch-lane candidate filter reads (NumRunning/NumWaiting),
// using the same makeSchedulerProvider fixture the other scheduler tests use.
func laneTestProvider(t *testing.T, reg *Registry, id, model string, decodeTPS float64, running, waiting int) *Provider {
	t.Helper()
	p := makeSchedulerProvider(t, reg, id, model, decodeTPS)
	p.mu.Lock()
	p.BackendCapacity.Slots[0].NumRunning = running
	p.BackendCapacity.Slots[0].NumWaiting = waiting
	p.mu.Unlock()
	return p
}

func batchLaneRequest(id string, lane Lane) *PendingRequest {
	return &PendingRequest{
		RequestID:             id,
		EstimatedPromptTokens: 500,
		RequestedMaxTokens:    256,
		Traits:                RequestTraits{Lane: lane},
	}
}

// planProviderIDs is the set of provider IDs a dispatch plan retained.
func planProviderIDs(plan *DispatchPlan) map[string]bool {
	ids := map[string]bool{}
	if plan == nil {
		return ids
	}
	for _, e := range plan.entries {
		ids[e.view.ProviderID] = true
	}
	return ids
}

// TestBatchRowsAllowedIsRouterCapMinusOne pins the batch allowance to the
// router's OWN per-model admission cap minus the row reserved for online, both
// with the quality-concurrency cap disabled (legacy flat cap) and enabled.
func TestBatchRowsAllowedIsRouterCapMinusOne(t *testing.T) {
	reg := New(testLogger())
	model := "batch-allowance-model"
	p := laneTestProvider(t, reg, "a", model, 40, 0, 0)

	want := effCap(reg, p, model) - 1
	if want < 0 {
		want = 0
	}
	if got := reg.batchRowsAllowed(p, model); got != want {
		t.Fatalf("batchRowsAllowed=%d, want %d (router cap %d minus one)",
			got, want, want+1)
	}

	// With the quality cap on, the allowance must track the quality cap — never
	// a parallel formula and never the legacy flat cap.
	enableQualityCap(t, reg, "")
	reg.mu.RLock()
	p.mu.Lock()
	capped := reg.effectiveMaxConcurrencyForModelResolvedLocked(p, model)
	p.mu.Unlock()
	reg.mu.RUnlock()
	wantCapped := capped - 1
	if wantCapped < 0 {
		wantCapped = 0
	}
	if got := reg.batchRowsAllowed(p, model); got != wantCapped {
		t.Fatalf("batchRowsAllowed with quality cap=%d, want %d (quality cap %d minus one)",
			got, wantCapped, capped)
	}
}

// TestBatchRowsAllowedFloorsAtZero: a pair whose router cap is 1 has no batch
// row at all — the single row it can run is the one reserved for online.
func TestBatchRowsAllowedFloorsAtZero(t *testing.T) {
	reg := New(testLogger())
	model := "batch-allowance-one-model"
	p := laneTestProvider(t, reg, "solo", model, 40, 0, 0)
	p.mu.Lock()
	p.BackendCapacity.Slots[0].MaxConcurrency = 1
	p.mu.Unlock()

	if got := reg.batchRowsAllowed(p, model); got != 0 {
		t.Fatalf("batchRowsAllowed=%d, want 0 for a cap-1 pair", got)
	}
}

// TestReserveLaneBatchOnlyTakesHeadroomSlots is the reservation-path contract:
// on LaneBatch a provider is a candidate only while its slot has no waiting row
// and its running count is below BatchRowsAllowed. On LaneOnline the same three
// providers are all candidates, so the filter is lane-scoped and changes nothing
// for existing traffic.
func TestReserveLaneBatchOnlyTakesHeadroomSlots(t *testing.T) {
	model := "batch-lane-model"

	newFleet := func(t *testing.T) (*Registry, int) {
		t.Helper()
		reg := New(testLogger())
		// A: fully idle — the only slot with batch headroom.
		laneTestProvider(t, reg, "A", model, 40, 0, 0)
		// B: one row already waiting. Admittable online, closed to batch: a
		// waiting row means the slot is oversubscribed right now.
		laneTestProvider(t, reg, "B", model, 40, 0, 1)
		// C: running exactly at its batch allowance. Admittable online (the
		// reserved online row is precisely what is left), closed to batch.
		c := laneTestProvider(t, reg, "C", model, 40, 0, 0)
		allowance := reg.batchRowsAllowed(c, model)
		if allowance < 1 {
			t.Fatalf("fixture: batch allowance %d leaves nothing to fill", allowance)
		}
		c.mu.Lock()
		c.BackendCapacity.Slots[0].NumRunning = allowance
		c.mu.Unlock()
		return reg, allowance
	}

	t.Run("batch lane takes only the idle provider", func(t *testing.T) {
		reg, allowance := newFleet(t)
		pr := batchLaneRequest("batch-1", LaneBatch)
		p, decision, plan := reg.ReserveProviderWithPlan(model, pr)
		if p == nil {
			t.Fatalf("batch reservation failed: %+v", decision)
		}
		if p.ID != "A" {
			t.Fatalf("batch winner=%q, want A (the only slot with headroom)", p.ID)
		}
		ids := planProviderIDs(plan)
		if ids["B"] {
			t.Fatal("plan retained B: a slot with a waiting row is not batch-eligible")
		}
		if ids["C"] {
			t.Fatalf("plan retained C: running %d == batch allowance %d is not batch-eligible",
				allowance, allowance)
		}
		if decision.GateRejections[GateBatchHeadroom] != 2 {
			t.Fatalf("batch_headroom gate rejections=%d, want 2 (B and C)",
				decision.GateRejections[GateBatchHeadroom])
		}
	})

	t.Run("online lane keeps every provider a candidate", func(t *testing.T) {
		reg, _ := newFleet(t)
		pr := batchLaneRequest("online-1", LaneOnline)
		p, decision, plan := reg.ReserveProviderWithPlan(model, pr)
		if p == nil {
			t.Fatalf("online reservation failed: %+v", decision)
		}
		if p.ID != "A" {
			t.Fatalf("online winner=%q, want A (lowest occupancy)", p.ID)
		}
		ids := planProviderIDs(plan)
		if !ids["B"] || !ids["C"] {
			t.Fatalf("online plan=%v, want both B and C retained", ids)
		}
		if decision.GateRejections[GateBatchHeadroom] != 0 {
			t.Fatalf("batch_headroom gate fired on the online lane: %d",
				decision.GateRejections[GateBatchHeadroom])
		}
	})

	t.Run("batch lane finds nothing when no slot has headroom", func(t *testing.T) {
		reg, _ := newFleet(t)
		// Close A too: one waiting row is enough.
		a := reg.GetProvider("A")
		a.mu.Lock()
		a.BackendCapacity.Slots[0].NumWaiting = 1
		a.mu.Unlock()

		pr := batchLaneRequest("batch-2", LaneBatch)
		p, decision, _ := reg.ReserveProviderWithPlan(model, pr)
		if p != nil {
			t.Fatalf("batch reserved %q with no headroom anywhere", p.ID)
		}
		if decision.CapacityRejections == 0 {
			t.Fatal("no-headroom batch scan reported zero capacity rejections")
		}
		// The same fleet still serves online traffic.
		onlinePR := batchLaneRequest("online-2", LaneOnline)
		if op, _, _ := reg.ReserveProviderWithPlan(model, onlinePR); op == nil {
			t.Fatal("online reservation failed on a fleet that is merely batch-closed")
		}
	})
}

// TestReserveLaneBatchDebitsLiveLoadWithinOneScan is the stale-row-count
// regression: the gate must read the occupancy number the online admission cap
// reads (pendingLoadForModelLocked, debited synchronously by addPendingLocked),
// not the heartbeat-derived NumRunning. Three batch reservations back to back
// against a slot whose batch allowance is 2, with NO heartbeat in between: the
// first two take the two batch rows, the third is refused. Gating on
// NumRunning alone would admit all three, because NumRunning does not move
// until the provider's next heartbeat.
func TestReserveLaneBatchDebitsLiveLoadWithinOneScan(t *testing.T) {
	reg := New(testLogger())
	model := "batch-live-load-model"
	p := laneTestProvider(t, reg, "solo", model, 40, 0, 0)
	// Router cap 3 → batch allowance 2, one row reserved for online.
	p.mu.Lock()
	p.BackendCapacity.Slots[0].MaxConcurrency = 3
	p.mu.Unlock()

	if got := reg.batchRowsAllowed(p, model); got != 2 {
		t.Fatalf("fixture: batchRowsAllowed=%d, want 2", got)
	}

	for i := 1; i <= 2; i++ {
		pr := batchLaneRequest(fmt.Sprintf("batch-live-%d", i), LaneBatch)
		got, decision, _ := reg.ReserveProviderWithPlan(model, pr)
		if got == nil {
			t.Fatalf("batch reservation %d failed: %+v", i, decision)
		}
	}

	// Third: the two batch rows are taken and no heartbeat has landed, so
	// NumRunning is still 0 while the live pending load is 2.
	p.mu.Lock()
	running := p.BackendCapacity.Slots[0].NumRunning
	p.mu.Unlock()
	if running != 0 {
		t.Fatalf("fixture: NumRunning=%d, want a stale 0 (no heartbeat)", running)
	}

	pr := batchLaneRequest("batch-live-3", LaneBatch)
	got, decision, _ := reg.ReserveProviderWithPlan(model, pr)
	if got != nil {
		t.Fatalf("third batch reservation took %q: the allowance of 2 was overrun "+
			"against a stale row count", got.ID)
	}
	if decision.GateRejections[GateBatchHeadroom] != 1 {
		t.Fatalf("batch_headroom gate rejections=%d, want 1",
			decision.GateRejections[GateBatchHeadroom])
	}

	// The reserved online row is still there: the same fleet serves online.
	if op, d, _ := reg.ReserveProviderWithPlan(model, batchLaneRequest("online-live", LaneOnline)); op == nil {
		t.Fatalf("online reservation failed on a merely batch-full slot: %+v", d)
	}
}

// TestReserveLaneBatchSkipsColdSlots: a provider that has the model on disk but
// not loaded (slotState "unknown") is a perfectly good ONLINE candidate — it
// pays the cold-load penalty and RecordWarmPoolColdDispatch is charged. Batch
// must never be the traffic that makes a provider load weights or evict another
// model, so the same provider is closed to the batch lane.
func TestReserveLaneBatchSkipsColdSlots(t *testing.T) {
	model := "batch-cold-model"

	newColdFleet := func(t *testing.T) *Registry {
		t.Helper()
		reg := New(testLogger())
		p := laneTestProvider(t, reg, "cold", model, 40, 0, 0)
		// Model advertised (Models list) but no slot for it: the snapshot's
		// slot state stays "unknown" — eligible, cold.
		p.mu.Lock()
		p.BackendCapacity.Slots = nil
		p.mu.Unlock()
		return reg
	}

	t.Run("batch skips the cold provider", func(t *testing.T) {
		reg := newColdFleet(t)
		p, decision, plan := reg.ReserveProviderWithPlan(model, batchLaneRequest("batch-cold", LaneBatch))
		if p != nil {
			t.Fatalf("batch reserved cold provider %q: it would force a model load", p.ID)
		}
		// No reservation and no retained alternate, so RecordWarmPoolColdDispatch
		// (scheduler.go, dispatch_plan.go — both keyed on !slotStateModelLoaded)
		// is never reached for a batch attempt.
		if ids := planProviderIDs(plan); len(ids) != 0 {
			t.Fatalf("batch plan retained %v: a cold slot is not batch-eligible", ids)
		}
		if decision.GateRejections[GateBatchHeadroom] != 1 {
			t.Fatalf("batch_headroom gate rejections=%d, want 1 (the cold slot)",
				decision.GateRejections[GateBatchHeadroom])
		}
	})

	t.Run("online still takes the cold provider", func(t *testing.T) {
		reg := newColdFleet(t)
		p, decision, _ := reg.ReserveProviderWithPlan(model, batchLaneRequest("online-cold", LaneOnline))
		if p == nil {
			t.Fatalf("online reservation failed on a cold-but-servable provider: %+v", decision)
		}
		if decision.GateRejections[GateBatchHeadroom] != 0 {
			t.Fatalf("batch_headroom gate fired on the online lane: %d",
				decision.GateRejections[GateBatchHeadroom])
		}
	})
}
