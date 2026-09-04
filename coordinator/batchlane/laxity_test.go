package batchlane

import (
	"testing"
	"time"
)

func TestLaxityHealthyBatchIsLowestPriority(t *testing.T) {
	now := time.Unix(0, 0)
	exp := now.Add(24 * time.Hour)
	// 1000 items left at 1/s needs 1000 s; there are 86400 s of window, so the
	// batch has 85400 s of slack — far more than the 6 h escalation horizon.
	l := Laxity(exp, now, 1000, 1.0)
	if want := 85400 * time.Second; l != want {
		t.Fatalf("laxity = %v, want %v", l, want)
	}
	if u := Urgency(l); u != 0 {
		t.Fatalf("urgency = %v, want 0", u)
	}
	if p := Priority(Urgency(l)); p != PriorityMax {
		t.Fatalf("priority = %d, want %d", p, PriorityMax)
	}
}

// A healthy batch never escalates however far through its window it is: the
// horizon is 6 h of SLACK, not a fraction of the 24 h completion window.
func TestLaxityLateInWindowButOnTrackStaysLowest(t *testing.T) {
	now := time.Unix(0, 0)
	exp := now.Add(12 * time.Hour) // half the window already elapsed
	l := Laxity(exp, now, 60, 1.0) // 60 s of work, 12 h of room
	if p := Priority(Urgency(l)); p != PriorityMax {
		t.Fatalf("priority = %d, want %d", p, PriorityMax)
	}
}

func TestLaxityLateBatchEscalates(t *testing.T) {
	now := time.Unix(0, 0)
	exp := now.Add(time.Hour)
	// 5000 items at 1/s needs 5000 s but only 3600 s remain: laxity is negative,
	// so the batch is projected to miss its window outright.
	l := Laxity(exp, now, 5000, 1.0)
	if l >= 0 {
		t.Fatalf("laxity = %v, want negative", l)
	}
	if u := Urgency(l); u != 1 {
		t.Fatalf("urgency = %v, want 1", u)
	}
	if p := Priority(Urgency(l)); p != PriorityMin {
		t.Fatalf("priority = %d, want %d", p, PriorityMin)
	}
}

func TestLaxityColdStartUsesSlackOnly(t *testing.T) {
	now := time.Unix(0, 0)
	exp := now.Add(3 * time.Hour)
	// No rate is known yet. Dividing by an epsilon here would make every fresh
	// batch maximally urgent, so an unknown rate means pure slack and the
	// caller substitutes the fleet rate or the floor.
	if l := Laxity(exp, now, 1000, 0); l != 3*time.Hour {
		t.Fatalf("laxity = %v, want %v", l, 3*time.Hour)
	}
	if l := Laxity(exp, now, 1000, -1); l != 3*time.Hour {
		t.Fatalf("laxity with a negative rate = %v, want %v", l, 3*time.Hour)
	}
}

func TestLaxityWithNothingLeftIsPureSlack(t *testing.T) {
	now := time.Unix(0, 0)
	exp := now.Add(90 * time.Minute)
	if l := Laxity(exp, now, 0, 0.5); l != 90*time.Minute {
		t.Fatalf("laxity = %v, want %v", l, 90*time.Minute)
	}
}

func TestUrgencyRampsLinearlyAcrossTheHorizon(t *testing.T) {
	if u := Urgency(EscalationHorizon); u != 0 {
		t.Fatalf("urgency at exactly the horizon = %v, want 0", u)
	}
	if u := Urgency(EscalationHorizon / 2); u != 0.5 {
		t.Fatalf("urgency at half the horizon = %v, want 0.5", u)
	}
	if u := Urgency(0); u != 1 {
		t.Fatalf("urgency at zero laxity = %v, want 1", u)
	}
	if u := Urgency(-time.Hour); u != 1 {
		t.Fatalf("urgency past the deadline = %v, want 1 (clamped)", u)
	}
	if u := Urgency(48 * time.Hour); u != 0 {
		t.Fatalf("urgency with a huge slack = %v, want 0 (clamped)", u)
	}
}

func TestPriorityIsBoundedAndHalfUp(t *testing.T) {
	if p := Priority(0); p != 100 {
		t.Fatalf("priority(0) = %d, want 100", p)
	}
	if p := Priority(1); p != 1 {
		t.Fatalf("priority(1) = %d, want 1", p)
	}
	if p := Priority(0.5); p != 51 {
		t.Fatalf("priority(0.5) = %d, want 51 (100 - 49.5 rounded half up)", p)
	}
	// Out-of-range urgency is clamped rather than escaping the [1, 100] band.
	if p := Priority(-3); p != 100 {
		t.Fatalf("priority(-3) = %d, want 100", p)
	}
	if p := Priority(9); p != 1 {
		t.Fatalf("priority(9) = %d, want 1", p)
	}
}

func TestObservedRateWindow(t *testing.T) {
	t0 := time.Unix(0, 0)
	var r ObservedRate

	if _, known := r.PerSec(t0); known {
		t.Fatal("an empty window must report unknown, not a rate of zero")
	}

	// 12 completions spread across the 60 s window -> 0.2 items/s.
	for i := 0; i < 12; i++ {
		r.Record(t0.Add(time.Duration(i) * 5 * time.Second))
	}
	now := t0.Add(59 * time.Second)
	rate, known := r.PerSec(now)
	if !known {
		t.Fatal("12 events in the window must report a known rate")
	}
	if rate != 0.2 {
		t.Fatalf("rate = %v, want 0.2", rate)
	}
}

func TestObservedRateEvictsPastTheWindow(t *testing.T) {
	t0 := time.Unix(0, 0)
	var r ObservedRate
	for i := 0; i < 12; i++ {
		r.Record(t0.Add(time.Duration(i) * 5 * time.Second))
	}
	// Two minutes later every sample is older than the window.
	if _, known := r.PerSec(t0.Add(2 * time.Minute)); known {
		t.Fatal("a fully evicted window must report unknown")
	}
	// And the eviction must actually release the samples, not just hide them.
	if n := r.Len(); n != 0 {
		t.Fatalf("retained %d events after eviction, want 0", n)
	}
}

func TestObservedRateHonoursACustomWindow(t *testing.T) {
	t0 := time.Unix(0, 0)
	r := ObservedRate{Window: 10 * time.Second}
	for i := 0; i < 5; i++ {
		r.Record(t0.Add(time.Duration(i) * time.Second))
	}
	rate, known := r.PerSec(t0.Add(9 * time.Second))
	if !known || rate != 0.5 {
		t.Fatalf("rate = %v (known %v), want 0.5", rate, known)
	}
}
