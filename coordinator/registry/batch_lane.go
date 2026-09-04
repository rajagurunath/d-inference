package registry

// batch_lane.go holds the registry side of the batch lane: how many concurrent
// rows a (provider, model) slot may hand to batch work.
//
// The number is NOT a second capacity formula. It is the router's own
// per-provider/per-model admission cap — effectiveMaxConcurrencyForModelResolvedLocked,
// which is min(the provider-reported slot cap, ceil(qualityConcurrency(solo,
// floor, k, base, fallback) × overcommit)) from concurrency_cap.go, with
// qualityConcurrency itself living in warm_pool_target.go and k =
// effectiveTPSLoadFactor — minus one. Deriving it from the live admission cap
// rather than re-deriving the quality batch here is what guarantees the two can
// never drift: whatever the router thinks the quality batch is (including the
// per-model solo-TPS resolution, the per-model overcommit override, the
// dedicated-model guard, and the kill switch that restores the legacy flat cap),
// batch gets that number minus one.
//
// Minus one, always: one row of every slot stays reserved for online traffic, so
// a batch request can never be the admission that pushes a slot to its cap and
// makes the NEXT online request wait. A pair whose cap is 1 therefore has no
// batch allowance at all.

// BatchRowsAllowed reports how many concurrent rows the batch lane may occupy on
// p's slot for model: the router's own quality-concurrency admission cap for the
// pair minus one, floored at zero.
//
// Exported so the batch dispatcher (coordinator/batchlane) and the reservation
// filter share one number rather than each computing their own. Takes r.mu (read)
// and then p.mu, the established lock order; callers that already hold both use
// batchRowsAllowedLocked.
func (r *Registry) BatchRowsAllowed(p *Provider, model string) int {
	if r == nil || p == nil {
		return 0
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	p.mu.Lock()
	defer p.mu.Unlock()
	return r.batchRowsAllowedLocked(p, model)
}

// batchRowsAllowedLocked is BatchRowsAllowed for a caller that already holds
// r.mu and p.mu (the routing snapshot path).
func (r *Registry) batchRowsAllowedLocked(p *Provider, model string) int {
	allowed := r.effectiveMaxConcurrencyForModelResolvedLocked(p, model) - 1
	if allowed < 0 {
		return 0
	}
	return allowed
}

// BatchSlot is a read-only copy of one provider·model slot's live capacity
// state plus that pair's batch row allowance — everything the batch
// dispatcher's per-slot controller reads, and nothing else.
//
// It exists so coordinator/batchlane never touches a live *Provider: every
// field is copied under the same r.mu/p.mu pair the reservation path takes, the
// established lock order, so the dispatcher can walk the fleet without holding
// a registry mutex (the same contract ProviderSnapshot gives the base-rewards
// engine).
type BatchSlot struct {
	ProviderID string
	Model      string
	// NumRunning / NumWaiting are the slot's live scheduler counters, whichever
	// lane the rows belong to.
	NumRunning int
	NumWaiting int
	// ObservedDecodeTPS is the provider's EWMA of measured per-request decode
	// throughput for the slot; 0 when unmeasured.
	ObservedDecodeTPS float64
	// ActiveTokenBudgetUsed / Max are the slot's KV budget. Max is 0 on a
	// provider that does not report one, in which case there is no KV signal.
	ActiveTokenBudgetUsed int64
	ActiveTokenBudgetMax  int64
	// BatchRowsAllowed is BatchRowsAllowed for the pair, resolved under the same
	// lock as the counters above so the allowance and the state it is compared
	// against are from the same instant.
	BatchRowsAllowed int
}

// BatchSlots returns a snapshot of every loaded slot serving model that the
// router would route to, or of every loaded routable slot in the fleet when
// model is empty. The empty form is what the batch dispatcher asks for: it
// decides only HOW MANY batch rows may be in flight fleet-wide, while WHERE
// each one lands stays the reservation path's decision.
func (r *Registry) BatchSlots(model string) []BatchSlot {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make([]BatchSlot, 0, len(r.providers))
	for _, p := range r.providers {
		p.mu.Lock()
		if (p.Status == StatusOnline || p.Status == StatusServing) && p.BackendCapacity != nil {
			for _, slot := range p.BackendCapacity.Slots {
				if model != "" && slot.Model != model {
					continue
				}
				if !slotStateModelLoaded(slot.State) {
					continue
				}
				if !r.providerServesRoutableModelLocked(p, slot.Model, false) {
					continue
				}
				out = append(out, BatchSlot{
					ProviderID:            p.ID,
					Model:                 slot.Model,
					NumRunning:            slot.NumRunning,
					NumWaiting:            slot.NumWaiting,
					ObservedDecodeTPS:     slot.ObservedDecodeTPS,
					ActiveTokenBudgetUsed: slot.ActiveTokenBudgetUsed,
					ActiveTokenBudgetMax:  slot.ActiveTokenBudgetMax,
					BatchRowsAllowed:      r.batchRowsAllowedLocked(p, slot.Model),
				})
			}
		}
		p.mu.Unlock()
	}
	return out
}

// QualityCapFloorTPS returns the per-request sustained-decode quality floor the
// admission cap is derived from (the warm pool's DecodeFloorTPS, 15 tok/s by
// default). The batch dispatcher compares a slot's observed decode rate against
// it before adding a row, so the lane backs off against the SAME floor the
// router sizes the quality batch with. 0 means the floor is disabled.
//
// Mirrors QualityCapOvercommit.
func (r *Registry) QualityCapFloorTPS() float64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.qualityCapFloorTPS
}
