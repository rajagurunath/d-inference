package registry

import (
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
	if got := reg.BatchRowsAllowed(p, model); got != want {
		t.Fatalf("BatchRowsAllowed=%d, want %d (router cap %d minus one)",
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
	if got := reg.BatchRowsAllowed(p, model); got != wantCapped {
		t.Fatalf("BatchRowsAllowed with quality cap=%d, want %d (quality cap %d minus one)",
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

	if got := reg.BatchRowsAllowed(p, model); got != 0 {
		t.Fatalf("BatchRowsAllowed=%d, want 0 for a cap-1 pair", got)
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
		allowance := reg.BatchRowsAllowed(c, model)
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
