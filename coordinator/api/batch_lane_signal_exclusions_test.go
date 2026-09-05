package api

import (
	"bytes"
	"log/slog"
	"testing"

	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/store"
)

// batchLanePending is capacityTestPending stamped with a lane, so the two halves
// of every exclusion test below differ in exactly one field.
func batchLanePending(model, providerID string, n int, lane registry.Lane) *registry.PendingRequest {
	pr := capacityTestPending(model, providerID, n)
	pr.Traits.Lane = lane
	return pr
}

// TestBatchTerminalsFeedNoCapacitySignal is the co-serving invariant on the
// capacity funnel: batch 503s must leave the pair's ONLINE routing state exactly
// as they found it. noteInferenceError is the single chokepoint that arms the
// capacity-reject cooldown, the gray-box budget clamp and the windowed
// capacity-503 rate — all three of which the reservation path scores providers
// on. An online control drives the identical terminals into an identical pair
// and trips all three, so the lane is demonstrably what changed the answer.
func TestBatchTerminalsFeedNoCapacitySignal(t *testing.T) {
	var logBuf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelWarn}))
	st := store.NewMemory(store.Config{AdminKey: "test-key"})
	reg := registry.New(logger)

	const model = "batch-signal-model"
	const rejectStr = "token_budget_exhausted: request exceeds active token budget"
	batchPair := makeRoutableProvider(t, reg, "p-batch", model)
	onlinePair := makeRoutableProvider(t, reg, "p-online", model)
	srv := NewServer(reg, st, ServerConfig{}, logger)

	for i := 0; i < 8; i++ {
		srv.noteInferenceError(batchPair.ID, batchLanePending(model, batchPair.ID, i, registry.LaneBatch),
			503, rejectStr, "", "")
	}

	if reg.CapacityCooldownActive(batchPair.ID, model) {
		t.Error("batch 503s armed the capacity-reject cooldown — the pair is now unroutable for online traffic")
	}
	if reg.BudgetClampActive(batchPair.ID, model) {
		t.Error("batch 503s armed the gray-box budget clamp")
	}
	if rate, samples := reg.CapacityRejectRate(batchPair.ID, model); rate != 0 || samples != 0 {
		t.Errorf("batch 503s moved the capacity reject rate to %.2f over %d samples, want 0/0", rate, samples)
	}
	if reg.InferenceErrorCooldownActive(batchPair.ID, model, "") {
		t.Error("batch 503s armed the inference-error cooldown")
	}

	// Online control: the same eight terminals on an identically-built pair.
	for i := 0; i < 8; i++ {
		srv.noteInferenceError(onlinePair.ID, batchLanePending(model, onlinePair.ID, i, registry.LaneOnline),
			503, rejectStr, "", "")
	}
	if !reg.CapacityCooldownActive(onlinePair.ID, model) {
		t.Fatal("online control did not trip the capacity cooldown — the fixture no longer proves the lane is the cause")
	}
	if rate, samples := reg.CapacityRejectRate(onlinePair.ID, model); rate == 0 || samples == 0 {
		t.Fatalf("online control left the capacity reject rate at %.2f/%d — fixture proves nothing", rate, samples)
	}
}

// TestBatchNeverLandsInTheORUptimeSeries: inference.request_outcome is the
// numerator and denominator of the uptime figure OpenRouter scores the online
// endpoint on. A batch attempt is placed on leftover headroom against a 120s
// budget, so neither its successes nor its failures describe traffic an online
// consumer sent. The online control emits on the same fixture.
func TestBatchNeverLandsInTheORUptimeSeries(t *testing.T) {
	srv, _ := testServer(t)
	collector := newUDPCollector(t)
	defer collector.Close()
	dd := newTestDD(t, collector)
	defer dd.Close()
	srv.SetDatadog(dd)

	const model = "batch-uptime-model"
	batch := &dispatchState{s: srv, model: model, lane: registry.LaneBatch}
	batch.recordDispatchedRequestOutcome(newUnknownKVBackendAttribution(), orClassProvider5xx)
	batch.recordDispatchedRequestOutcome(newUnknownKVBackendAttribution(), orClassSuccess)

	_ = dd.Statsd.Flush()
	if outcomes := findMetrics(collector.drain(), metricRequestOutcome); len(outcomes) != 0 {
		t.Fatalf("batch emitted %d OR-uptime samples: %v", len(outcomes), outcomes)
	}

	online := &dispatchState{s: srv, model: model, lane: registry.LaneOnline}
	online.recordDispatchedRequestOutcome(newUnknownKVBackendAttribution(), orClassProvider5xx)
	_ = dd.Statsd.Flush()
	if outcomes := findMetrics(collector.drain(), metricRequestOutcome); len(outcomes) != 1 {
		t.Fatalf("online control emitted %d OR-uptime samples, want 1: %v", len(outcomes), outcomes)
	}
}

// TestBatchLaneNeverHedgesUnderPreferPolicy is the skipBackup-clobber
// regression. The lane suppression and the prefer-policy suppression are
// independent reasons to skip; assigning the second over the first handed a
// batch request its speculative backup back whenever the prefer primary was NOT
// the caller's own machine.
//
// The server is built WITHOUT a hedge governor on purpose: with no governor,
// runSpeculative never consults tryAcquireBackupHedge, so this predicate is the
// only thing standing between a batch item and a second billed attempt.
func TestBatchLaneNeverHedgesUnderPreferPolicy(t *testing.T) {
	srv, _ := testServer(t)
	srv.hedgeGov = nil

	const model = "batch-prefer-model"
	logger := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelError}))
	reg := registry.New(logger)
	// A PUBLIC provider: not the prefer owner, so the prefer clause is false and
	// would overwrite the lane's true with it.
	provider := makeRoutableProvider(t, reg, "p-public", model)
	provider.Mu().Lock()
	provider.AccountID = "someone-else"
	provider.Mu().Unlock()

	batch := &dispatchState{
		s:      srv,
		model:  model,
		lane:   registry.LaneBatch,
		policy: selfRoutePolicy{prefer: true, ownerAccountID: "acct-owner"},
	}
	if !batch.skipSpeculativeBackup(provider) {
		t.Fatal("batch lane launched a speculative backup: the prefer policy overwrote the lane suppression")
	}

	// And with prefer off entirely, the lane still suppresses.
	batchNoPolicy := &dispatchState{s: srv, model: model, lane: registry.LaneBatch}
	if !batchNoPolicy.skipSpeculativeBackup(provider) {
		t.Fatal("batch lane launched a speculative backup with no prefer policy at all")
	}

	// Online control on the same public provider: the hedge is NOT suppressed,
	// so the batch answers above are the lane and not the fixture.
	online := &dispatchState{
		s:      srv,
		model:  model,
		lane:   registry.LaneOnline,
		policy: selfRoutePolicy{prefer: true, ownerAccountID: "acct-owner"},
	}
	if online.skipSpeculativeBackup(provider) {
		t.Fatal("online prefer request to a PUBLIC provider was denied its speculative backup")
	}

	// The prefer rule itself is unchanged: an owner-served primary still skips.
	owned := makeRoutableProvider(t, reg, "p-owned", model)
	owned.Mu().Lock()
	owned.AccountID = "acct-owner"
	owned.Mu().Unlock()
	if !online.skipSpeculativeBackup(owned) {
		t.Fatal("prefer request served by the caller's OWN machine was raced by a paid public backup")
	}
}

// TestBatchSuccessesFeedNoCapacitySignal is the other half of the co-serving
// invariant, and the one that was missing: batch must not RE-ADMIT a pair that
// online traffic quarantined. noteInferenceSuccess clears the inference-error
// strike state, the capacity-reject cooldown and its re-trip backoff, and closes
// the node-health breaker — every one of them a gate the ONLINE reservation path
// respects. A batch attempt runs on leftover headroom, so its success says
// nothing about whether the pair can carry the traffic that shut it out.
//
// Both halves quarantine an identically-built pair with the same eight ONLINE
// terminals and then differ in exactly one field: the lane of the completion.
func TestBatchSuccessesFeedNoCapacitySignal(t *testing.T) {
	var logBuf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelWarn}))
	st := store.NewMemory(store.Config{AdminKey: "test-key"})
	reg := registry.New(logger)

	const model = "batch-success-model"
	const rejectStr = "token_budget_exhausted: request exceeds active token budget"
	batchPair := makeRoutableProvider(t, reg, "p-batch", model)
	onlinePair := makeRoutableProvider(t, reg, "p-online", model)
	srv := NewServer(reg, st, ServerConfig{}, logger)

	// Quarantine both pairs with ONLINE terminals, so what differs below is the
	// lane of the completion and nothing else.
	quarantine := func(providerID string) {
		for i := 0; i < 8; i++ {
			srv.noteInferenceError(providerID, batchLanePending(model, providerID, i, registry.LaneOnline),
				503, rejectStr, "", "")
			// A second, non-capacity terminal so the inference-error strike
			// state and the node-health breaker are armed as well: the capacity
			// funnel and the error funnel are separate trackers and
			// noteInferenceSuccess clears BOTH.
			srv.noteInferenceError(providerID, batchLanePending(model, providerID, i, registry.LaneOnline),
				500, "provider crashed while generating", "", "")
		}
		if !reg.CapacityCooldownActive(providerID, model) {
			t.Fatalf("fixture: %s kept no capacity cooldown, so a clearing test proves nothing", providerID)
		}
		if !reg.InferenceErrorCooldownActive(providerID, model, registry.RequestTraits{}.CooldownShape()) {
			t.Fatalf("fixture: %s kept no inference-error cooldown, so a clearing test proves nothing", providerID)
		}
		if !reg.ProviderBreakerOpen(providerID) {
			t.Fatalf("fixture: %s kept no open node-health breaker, so a clearing test proves nothing", providerID)
		}
	}
	quarantine(batchPair.ID)
	quarantine(onlinePair.ID)

	batchDone := batchLanePending(model, batchPair.ID, 99, registry.LaneBatch)
	batchDone.ProviderID = batchPair.ID
	srv.noteInferenceSuccess(batchDone)

	if !reg.CapacityCooldownActive(batchPair.ID, model) {
		t.Error("a batch success cleared the capacity-reject cooldown — the pair is routable online again")
	}
	if !reg.InferenceErrorCooldownActive(batchPair.ID, model, batchDone.Traits.CooldownShape()) {
		t.Error("a batch success cleared the inference-error cooldown")
	}
	if reg.ProviderBreakerOpen(batchPair.ID) != true {
		t.Error("a batch success closed the node-health breaker")
	}

	// Online control: the identical completion on the identically-quarantined
	// pair DOES re-admit it, so the assertions above are the lane and not a
	// cooldown that simply never clears.
	onlineDone := batchLanePending(model, onlinePair.ID, 99, registry.LaneOnline)
	onlineDone.ProviderID = onlinePair.ID
	srv.noteInferenceSuccess(onlineDone)
	if reg.CapacityCooldownActive(onlinePair.ID, model) {
		t.Fatal("online control did not clear the capacity cooldown — the fixture no longer proves the lane is the cause")
	}
	if reg.InferenceErrorCooldownActive(onlinePair.ID, model, onlineDone.Traits.CooldownShape()) {
		t.Fatal("online control did not clear the inference-error cooldown — fixture proves nothing")
	}
}
