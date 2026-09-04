// Package batchlane is the Tidal batch dispatcher: the 1 Hz control loop that
// fills provider slots the online quality cap is leaving empty with 24-hour
// batch work (docs/design/tidal-batch-lane.md §3.4).
//
// The package imports store and registry, never api — api imports batchlane.
// The dispatch funnel therefore reaches it as a DispatchFn the api layer wires
// to (*api.Server).DispatchBatchItem.
//
// Every stateful piece here takes its `now` from the caller. Nothing in this
// package calls time.Now(), so the whole control loop is testable without
// sleeping and a tick's decisions are a pure function of (state, signals, now).
package batchlane

import "time"

// control.go holds the pure per-slot control law: an EWMA to smooth the noisy
// heartbeat signals, the AIMD controller over the allowed batch in-flight
// count, and the token bucket behind the deadline progress floor.

// Control constants, verbatim from docs/design/tidal-batch-lane.md §3.4.
const (
	// EWMAAlpha is the smoothing factor applied to the per-slot decode rate and
	// KV pressure. Half the weight on the newest sample: fast enough to see a
	// load spike within a couple of ticks, slow enough that one noisy heartbeat
	// does not collapse the target.
	EWMAAlpha = 0.5
	// KVHigh is the KV-pressure high watermark. Above it the slot is close
	// enough to eviction that batch must get out of the way immediately.
	KVHigh = 0.85
	// KVLow is the KV-pressure low watermark. Below it there is provably room
	// for one more row. The gap to KVHigh is the hysteresis that stops the
	// controller from thrashing between increase and decrease every tick.
	KVLow = 0.70
)

// EWMA is an exponentially weighted moving average seeded by its first sample,
// so a fresh slot reports the sample itself rather than a value dragged toward
// zero. The zero value is ready to use and smooths at EWMAAlpha.
type EWMA struct {
	// Alpha is the weight on the newest sample, in (0, 1]. <= 0 means EWMAAlpha.
	Alpha float64
	// Value is the current average; meaningful only once Initialized.
	Value float64

	init bool
}

// Observe folds one sample in and returns the new average.
func (e *EWMA) Observe(sample float64) float64 {
	alpha := e.Alpha
	if alpha <= 0 || alpha > 1 {
		alpha = EWMAAlpha
	}
	if !e.init {
		e.init = true
		e.Value = sample
		return e.Value
	}
	e.Value = alpha*sample + (1-alpha)*e.Value
	return e.Value
}

// Initialized reports whether any sample has been folded in yet.
func (e *EWMA) Initialized() bool { return e.init }

// SlotSignal is one provider·slot's live state for a single control step. It is
// assembled by a RegistryView from the provider's heartbeat capacity report;
// MaxPerSlot is (*registry.Registry).BatchRowsAllowed for the pair, i.e. the
// router's own admission cap minus the row reserved for online traffic.
type SlotSignal struct {
	// Waiting is the slot's NumWaiting — rows queued in the provider's own
	// scheduler, whichever lane they belong to. Any waiting row at all means
	// the slot is oversubscribed right now.
	Waiting int
	// DecodeTPS is the smoothed observed per-request decode rate (tok/s).
	DecodeTPS float64
	// DecodeFloor is the router's quality floor for that rate (tok/s), from
	// registry's quality-concurrency cap. <= 0 disables the trigger.
	DecodeFloor float64
	// KV is ActiveTokenBudgetUsed / ActiveTokenBudgetMax, smoothed, in [0, 1].
	// Meaningful only when KVKnown.
	KV float64
	// KVKnown reports whether the provider published a token budget at all. A
	// slot that publishes none has no KV signal: the controller must drop the
	// KV terms for it rather than substitute a number. Substituting 0 reads as
	// "idle" and grows the target forever; substituting a hold-band value pins
	// the target wherever it happens to be, which for a fresh slot is 0 — the
	// lane would never start on a fleet that reports no budget.
	KVKnown bool
	// Running is the slot's NumRunning. Reported for logging and for the
	// dispatcher's budget arithmetic; the control law does not read it.
	Running int
	// MaxPerSlot is the batch row allowance for the pair. Zero means the slot
	// has no batch row at all and the target is pinned to zero.
	MaxPerSlot int
}

// AIMDConfig are the controller's watermarks. The zero value means the spec
// defaults (KVHigh / KVLow); MaxPerSlot is an optional global ceiling applied on
// top of the per-slot allowance, for an operator who wants to cap the lane
// below what the router would allow.
type AIMDConfig struct {
	KVHigh     float64
	KVLow      float64
	MaxPerSlot int
}

// AIMD tracks the allowed batch in-flight count for one slot: multiplicative
// decrease the moment online traffic or KV pressure shows up, additive increase
// while there is provable room, hold in between. It starts at zero, so a slot
// the dispatcher has never seen contributes nothing until it has been observed
// healthy for a tick.
//
// Floor is the deadline-escalation guarantee (design §3.4 step 3): once a batch
// crosses floorUrgency it is granted a minimum concurrency so a busy online
// tenant cannot starve it past its 24-hour window.
type AIMD struct {
	Target int
	Floor  int

	cfg AIMDConfig
}

// NewAIMD returns a controller with the given watermarks, starting at target 0.
func NewAIMD(cfg AIMDConfig) *AIMD { return &AIMD{cfg: cfg} }

// Update runs one control step against a fresh signal and returns the new
// target. It is a pure function of (state, sig) — no clock, no I/O.
//
// A slot with no KV signal (KVKnown false) is driven by the remaining terms
// alone: it decreases on a waiting row or a decode rate under the floor, and
// increases otherwise. Those two are enough backpressure on their own — a slot
// running out of KV starts queueing, and NumWaiting is not smoothed — so the
// lane still starts and still backs off on a fleet that publishes no budget.
func (a *AIMD) Update(sig SlotSignal) int {
	kvHigh, kvLow := a.watermarks()

	switch {
	case sig.Waiting > 0 ||
		// A slot that reports no decode rate at all is UNMEASURED, not slow.
		// Reading 0 as "below the floor" would pin every fresh provider's
		// target at the floor and the lane would never start.
		(sig.DecodeFloor > 0 && sig.DecodeTPS > 0 && sig.DecodeTPS < sig.DecodeFloor) ||
		(sig.KVKnown && sig.KV > kvHigh):
		a.Target /= 2 // multiplicative decrease
	case !sig.KVKnown || sig.KV < kvLow:
		a.Target++ // additive increase
	}
	a.Target = a.clamp(a.Target, sig.MaxPerSlot)
	return a.Target
}

// SetFloor sets the deadline-driven minimum concurrency. Raising it takes
// effect immediately; lowering it never drops the current target, which the
// control law walks back down on its own so a released floor cannot produce a
// step change.
func (a *AIMD) SetFloor(f int) {
	if f < 0 {
		f = 0
	}
	a.Floor = f
	if a.Target < a.Floor {
		a.Target = a.Floor
	}
}

// clamp bounds n to [Floor, max], where max is the slot's batch row allowance
// narrowed by any configured global ceiling. A slot with no allowance is pinned
// to zero — the floor cannot manufacture a row the router would not admit.
func (a *AIMD) clamp(n, maxPerSlot int) int {
	if maxPerSlot < 0 {
		maxPerSlot = 0
	}
	if a.cfg.MaxPerSlot > 0 && a.cfg.MaxPerSlot < maxPerSlot {
		maxPerSlot = a.cfg.MaxPerSlot
	}
	if n > maxPerSlot {
		n = maxPerSlot
	}
	if n < a.Floor {
		n = a.Floor
	}
	if n > maxPerSlot {
		n = maxPerSlot
	}
	if n < 0 {
		n = 0
	}
	return n
}

func (a *AIMD) watermarks() (high, low float64) {
	high, low = a.cfg.KVHigh, a.cfg.KVLow
	if high <= 0 {
		high = KVHigh
	}
	if low <= 0 {
		low = KVLow
	}
	return high, low
}

// TokenBucket is the deadline progress floor's rate limiter: it grants an
// urgent batch one in-flight item every 1/Rate seconds even when the AIMD
// target is zero, so a batch that is going to miss its window is never starved
// all the way to expiry. The zero value holds no tokens; a bucket starts full
// at its first TryTake so an urgent batch makes progress on the tick it becomes
// urgent rather than 1/Rate seconds later.
//
// Deterministic by construction: the caller passes the timestamp.
type TokenBucket struct {
	// Rate is the refill rate in items per second (floorItemsPerSec = 0.2).
	Rate float64
	// Capacity is the burst size, in items. <= 0 means 1.
	Capacity float64

	tokens float64
	last   time.Time
}

// TryTake refills for the elapsed time and takes one token if one is available.
func (b *TokenBucket) TryTake(now time.Time) bool {
	capacity := b.Capacity
	if capacity <= 0 {
		capacity = 1
	}
	if b.last.IsZero() {
		b.last = now
		b.tokens = capacity
	} else if elapsed := now.Sub(b.last); elapsed > 0 {
		// A backwards clock mints nothing and does not move `last`, so the
		// tokens the bucket already holds are never silently reset.
		b.last = now
		b.tokens += elapsed.Seconds() * b.Rate
		if b.tokens > capacity {
			b.tokens = capacity
		}
	}
	// The epsilon absorbs the float error in `elapsed * Rate` so a bucket that
	// has refilled for exactly 1/Rate seconds is not one ulp short of a token.
	if b.tokens+1e-9 < 1 {
		return false
	}
	b.tokens--
	if b.tokens < 0 {
		b.tokens = 0
	}
	return true
}
