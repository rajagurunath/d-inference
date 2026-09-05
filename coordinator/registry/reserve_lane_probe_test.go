package registry

import (
	"testing"
	"time"
)

// capacityProbeClaimedFor reports whether the pair's half-open probe has been
// claimed. Keyed through faultKeyLocked, exactly as the production maps are.
func capacityProbeClaimedFor(r *Registry, providerID, modelID string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.capacityCooldowns[capacityRejectKey{ProviderID: r.faultKeyLocked(providerID), ModelID: modelID}]
	return ok && !e.probeAt.IsZero()
}

// halfOpenCapacityCooldown trips the pair's capacity cooldown and rewinds its
// TTL into the past, leaving it half-open with the single probe unclaimed —
// the state a pair is in the instant its cooldown lapses.
func halfOpenCapacityCooldown(t *testing.T, r *Registry, providerID, modelID string) {
	t.Helper()
	for i := 0; i < r.capacityCooldownCfg.Threshold; i++ {
		r.RecordCapacityReject(providerID, modelID)
	}
	if !r.CapacityCooldownActive(providerID, modelID) {
		t.Fatalf("fixture: cooldown for %s/%s did not trip", providerID, modelID)
	}
	r.mu.Lock()
	e, ok := r.capacityCooldowns[capacityRejectKey{ProviderID: r.faultKeyLocked(providerID), ModelID: modelID}]
	if ok {
		e.expiry = time.Now().Add(-time.Second)
		e.probeAt = time.Time{}
	}
	r.mu.Unlock()
	if !ok {
		t.Fatalf("fixture: no cooldown entry for %s/%s", providerID, modelID)
	}
	if r.CapacityCooldownActive(providerID, modelID) {
		t.Fatalf("fixture: %s/%s not half-open after the TTL lapsed", providerID, modelID)
	}
}

// TestReserveLaneBatchLeavesCapacityProbeUnclaimed is invariant 4 on the
// half-open capacity probe: the probe is the router's ONE re-admission test for
// a pair whose cooldown just lapsed, and every batch terminal returns early
// from the outcome sites that resolve it. A batch reservation that claimed it
// would therefore close the pair to online traffic for the whole
// capacityProbeOutcomeWindow and never resolve it. Batch must leave the probe
// available; the next online reservation must be the one that claims it.
func TestReserveLaneBatchLeavesCapacityProbeUnclaimed(t *testing.T) {
	reg := New(testLogger())
	model := "batch-probe-model"
	p := laneTestProvider(t, reg, "solo", model, 40, 0, 0)
	// Router cap 4 → batch allowance 3, so neither reservation below runs out
	// of rows and the probe is the only thing under test.
	p.mu.Lock()
	p.BackendCapacity.Slots[0].MaxConcurrency = 4
	p.mu.Unlock()

	halfOpenCapacityCooldown(t, reg, "solo", model)

	got, decision, _ := reg.ReserveProviderWithPlan(model, batchLaneRequest("batch-probe", LaneBatch))
	if got == nil {
		t.Fatalf("batch reservation failed on a half-open pair: %+v", decision)
	}
	if capacityProbeClaimedFor(reg, "solo", model) {
		t.Fatal("batch reservation claimed the half-open capacity probe")
	}
	if reg.CapacityCooldownActive("solo", model) {
		t.Fatal("pair closed to online traffic by a batch reservation")
	}

	// The very next online reservation is the real probe: it claims the slot
	// and the gate closes behind it until the outcome lands.
	got, decision, _ = reg.ReserveProviderWithPlan(model, batchLaneRequest("online-probe", LaneOnline))
	if got == nil {
		t.Fatalf("online reservation failed on an unclaimed half-open pair: %+v", decision)
	}
	if !capacityProbeClaimedFor(reg, "solo", model) {
		t.Fatal("online reservation did not claim the half-open capacity probe")
	}
	if !reg.CapacityCooldownActive("solo", model) {
		t.Fatal("gate did not close behind the claimed probe")
	}
	// And the claim is exclusive: a second online request finds the pair shut.
	if second, _, _ := reg.ReserveProviderWithPlan(model, batchLaneRequest("online-probe-2", LaneOnline)); second != nil {
		t.Fatal("a second online reservation got through a claimed probe")
	}
}

// TestReserveNextFromPlanLaneBatchLeavesCapacityProbeUnclaimed is the same
// contract on the plan-consumption path (retry/hedge lane), which commits
// reservations through its own copy of the claim sequence.
func TestReserveNextFromPlanLaneBatchLeavesCapacityProbeUnclaimed(t *testing.T) {
	reg := New(testLogger())
	model := "batch-probe-plan-model"
	for _, id := range []string{"A", "B", "C"} {
		p := laneTestProvider(t, reg, id, model, 40, 0, 0)
		p.mu.Lock()
		p.BackendCapacity.Slots[0].MaxConcurrency = 4
		p.mu.Unlock()
		halfOpenCapacityCooldown(t, reg, id, model)
	}

	// One scan builds the plan; the two alternates it retains are consumed
	// below, one per lane. Building it on the batch lane keeps the winner's own
	// probe out of the picture.
	winner, decision, plan := reg.ReserveProviderWithPlan(model, batchLaneRequest("batch-plan-0", LaneBatch))
	if winner == nil {
		t.Fatalf("batch reservation failed: %+v", decision)
	}

	batchAlt, decision, skips := reg.ReserveNextFromPlan(
		batchLaneRequest("batch-plan-1", LaneBatch), plan, winner.ID)
	if batchAlt == nil {
		t.Fatalf("plan batch reservation failed: %+v skips=%v", decision, skips)
	}
	if capacityProbeClaimedFor(reg, batchAlt.ID, model) {
		t.Fatalf("plan batch reservation claimed %s's half-open capacity probe", batchAlt.ID)
	}

	onlineAlt, decision, skips := reg.ReserveNextFromPlan(
		batchLaneRequest("online-plan-1", LaneOnline), plan, winner.ID)
	if onlineAlt == nil {
		t.Fatalf("plan online reservation failed: %+v skips=%v", decision, skips)
	}
	if onlineAlt.ID == batchAlt.ID {
		t.Fatalf("fixture: both plan reservations landed on %s", onlineAlt.ID)
	}
	if !capacityProbeClaimedFor(reg, onlineAlt.ID, model) {
		t.Fatalf("plan online reservation did not claim %s's half-open capacity probe", onlineAlt.ID)
	}
	if capacityProbeClaimedFor(reg, batchAlt.ID, model) {
		t.Fatalf("%s's probe was claimed after the fact", batchAlt.ID)
	}
}
