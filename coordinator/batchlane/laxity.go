package batchlane

// laxity.go holds the pure deadline math behind batch priority
// (docs/design/tidal-batch-lane.md §3.4 step 3), Least-Laxity-First:
//
//	laxity   = (expires_at − now) − remaining_items / rate
//	urgency  = clamp(1 − laxity / EscalationHorizon, 0, 1)
//	priority = round(PriorityMax − urgency × (PriorityMax − PriorityMin))
//
// The horizon is NOT the completion window. Dividing by the window would make
// slack itself escalating: a perfectly healthy batch twelve hours from its
// deadline with a minute of work left would show urgency 0.5 and start taking
// capacity from online traffic for no reason. Measured against a horizon,
// laxity ≥ 6 h means urgency 0 — an on-track batch never escalates, however
// far through its window it is — and the ramp to the highest priority happens
// over the last 6 h of slack, the only region where escalating buys anything.

import (
	"math"
	"time"
)

// Deadline constants, verbatim from docs/design/tidal-batch-lane.md §3.4.
const (
	// EscalationHorizon is the slack over which urgency ramps from 0 to 1.
	EscalationHorizon = 6 * time.Hour
	// PriorityMax is the priority of an on-track batch: the lowest there is,
	// and the value every batch sits at until its slack runs low.
	PriorityMax = 100
	// PriorityMin is the priority of a batch projected to miss its window.
	PriorityMin = 1
	// FloorUrgency is the urgency past which a batch is guaranteed a progress
	// floor of one in-flight item through its own token bucket, so a busy
	// online tenant can never starve it all the way to expiry.
	FloorUrgency = 0.9
	// FloorItemsPerSec is that token bucket's refill rate.
	FloorItemsPerSec = 0.2
	// ObservedRateWindow is the trailing window ObservedRate measures over.
	ObservedRateWindow = 60 * time.Second
)

// Laxity is the slack left after the projected drain time. Negative laxity
// means the batch is projected to miss its completion window at the current
// rate.
//
// An unknown rate (itemsPerSec <= 0) yields pure slack rather than an infinite
// drain time: a batch that has completed nothing yet has no evidence it is
// slow, and treating cold start as maximally urgent would let every freshly
// created batch preempt the fleet. The caller substitutes the fleet-wide
// completion rate, and then FloorItemsPerSec, before giving up on a rate.
func Laxity(expiresAt, now time.Time, remaining int, itemsPerSec float64) time.Duration {
	slack := expiresAt.Sub(now)
	if remaining <= 0 || itemsPerSec <= 0 {
		return slack
	}
	drainSecs := float64(remaining) / itemsPerSec
	if drainSecs >= math.MaxInt64/float64(time.Second) {
		// A rate small enough to overflow the duration is, for every purpose
		// the caller has, "will never finish".
		return time.Duration(math.MinInt64)
	}
	return slack - time.Duration(drainSecs*float64(time.Second))
}

// Urgency maps laxity onto [0, 1]: 0 while a full escalation horizon of slack
// remains, 1 the moment the projected finish time reaches the deadline.
func Urgency(laxity time.Duration) float64 {
	u := 1 - laxity.Seconds()/EscalationHorizon.Seconds()
	return clamp01(u)
}

// Priority maps urgency onto the [PriorityMin, PriorityMax] band, lower being
// scheduled sooner. Rounding is half-up so band edges are predictable.
func Priority(urgency float64) int {
	u := clamp01(urgency)
	span := float64(PriorityMax - PriorityMin)
	p := int(math.Floor(float64(PriorityMax) - u*span + 0.5))
	if p > PriorityMax {
		p = PriorityMax
	}
	if p < PriorityMin {
		p = PriorityMin
	}
	return p
}

func clamp01(v float64) float64 {
	if math.IsNaN(v) {
		return 1 // fail safe in the escalate direction
	}
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// ObservedRate is a sliding-window completion counter, one per batch. The zero
// value measures over ObservedRateWindow.
//
// The divisor is the whole window, not the observed span, so a batch that has
// completed three items in its first two seconds reports 0.05/s rather than
// 1.5/s. Under-reporting is the safe direction: it lengthens the projected
// drain time, which raises urgency, which gets the batch more capacity.
//
// Not safe for concurrent use; the dispatcher owns one per batch and touches
// them only from its tick.
type ObservedRate struct {
	// Window is the trailing window. <= 0 means ObservedRateWindow.
	Window time.Duration

	events []time.Time
}

// Record notes one completion at now.
func (r *ObservedRate) Record(now time.Time) {
	r.events = append(r.events, now)
}

// PerSec returns the completion rate over the window. known is false when
// nothing completed in it, so the caller can fall back to the fleet rate rather
// than reading an empty window as a rate of zero.
func (r *ObservedRate) PerSec(now time.Time) (itemsPerSec float64, known bool) {
	r.evict(now)
	if len(r.events) == 0 {
		return 0, false
	}
	return float64(len(r.events)) / r.window().Seconds(), true
}

// Len is the number of samples still inside the window as of the last eviction.
func (r *ObservedRate) Len() int { return len(r.events) }

func (r *ObservedRate) evict(now time.Time) {
	cutoff := now.Add(-r.window())
	keep := 0
	for _, ts := range r.events {
		if ts.After(cutoff) {
			break
		}
		keep++
	}
	if keep == 0 {
		return
	}
	// Copy down rather than reslicing so the evicted timestamps are released
	// and a long-lived batch's backing array cannot grow without bound.
	rest := r.events[keep:]
	r.events = append(r.events[:0], rest...)
}

func (r *ObservedRate) window() time.Duration {
	if r.Window <= 0 {
		return ObservedRateWindow
	}
	return r.Window
}
