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
