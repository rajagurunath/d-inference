package batchlane

import (
	"testing"
	"time"
)

// healthySignal is a slot with plenty of room: nothing waiting, decode well
// above the floor, KV below the low watermark. Tests mutate the one field they
// are about so the trigger under test is the only thing that differs.
func healthySignal(maxPerSlot int) SlotSignal {
	return SlotSignal{
		Waiting:     0,
		DecodeTPS:   40,
		DecodeFloor: 15,
		KV:          0.20,
		Running:     0,
		MaxPerSlot:  maxPerSlot,
	}
}

func TestAIMDHalvesOnAnyWaiting(t *testing.T) {
	a := AIMD{Target: 8}
	sig := healthySignal(8)
	sig.Waiting = 1
	if got := a.Update(sig); got != 4 {
		t.Fatalf("target after one waiting row = %d, want 4", got)
	}
	if a.Target != 4 {
		t.Fatalf("Target field = %d, want 4", a.Target)
	}
}

// The decrease rule fires on ANY waiting row, whichever lane it belongs to —
// the dispatcher never subtracts its own in-flight count (plan Global
// Constraints; unlike the Tidal reference, the coordinator's NumWaiting is
// provider-side queueing that batch admission is supposed to have avoided).
func TestAIMDHalvesOnWaitingEvenWithBatchInFlight(t *testing.T) {
	a := AIMD{Target: 8}
	sig := healthySignal(8)
	sig.Waiting = 1
	sig.Running = 6
	if got := a.Update(sig); got != 4 {
		t.Fatalf("target = %d, want 4 (waiting must not be discounted by in-flight)", got)
	}
}

func TestAIMDHalvesBelowDecodeFloor(t *testing.T) {
	a := AIMD{Target: 8}
	sig := healthySignal(8)
	sig.DecodeTPS = 12 // below the 15 tok/s floor
	if got := a.Update(sig); got != 4 {
		t.Fatalf("target below the decode floor = %d, want 4", got)
	}
}

func TestAIMDHalvesOnKVHigh(t *testing.T) {
	a := AIMD{Target: 8}
	sig := healthySignal(8)
	sig.KV = 0.90
	if got := a.Update(sig); got != 4 {
		t.Fatalf("target at KV 0.90 = %d, want 4", got)
	}
}

func TestAIMDIncrementsWhenKVLow(t *testing.T) {
	a := AIMD{Target: 3}
	if got := a.Update(healthySignal(8)); got != 4 {
		t.Fatalf("target at KV 0.20 = %d, want 4", got)
	}
}

func TestAIMDHoldsBetweenWatermarks(t *testing.T) {
	a := AIMD{Target: 5}
	sig := healthySignal(8)
	sig.KV = 0.78 // between kvLow 0.70 and kvHigh 0.85
	if got := a.Update(sig); got != 5 {
		t.Fatalf("target in the hysteresis band = %d, want 5 (unchanged)", got)
	}
}

func TestAIMDNeverBelowFloorOrAboveMax(t *testing.T) {
	// Decrease cannot go below the floor.
	a := AIMD{Target: 4, Floor: 3}
	sig := healthySignal(8)
	sig.KV = 0.90
	if got := a.Update(sig); got != 3 {
		t.Fatalf("target = %d, want the floor 3", got)
	}
	if got := a.Update(sig); got != 3 {
		t.Fatalf("target after a second decrease = %d, want the floor 3", got)
	}

	// Increase cannot go above the slot's batch row allowance.
	b := AIMD{Target: 2}
	if got := b.Update(healthySignal(2)); got != 2 {
		t.Fatalf("target = %d, want MaxPerSlot 2", got)
	}

	// A pair with no batch allowance at all is pinned to zero, floor included.
	c := AIMD{Target: 4, Floor: 2}
	if got := c.Update(healthySignal(0)); got != 0 {
		t.Fatalf("target on a zero-allowance slot = %d, want 0", got)
	}
}

func TestAIMDSetFloorRaisesTargetImmediately(t *testing.T) {
	a := AIMD{Target: 0}
	a.SetFloor(2)
	if a.Target != 2 {
		t.Fatalf("Target after SetFloor(2) = %d, want 2", a.Target)
	}
	// Lowering the floor never drops the target; the control law walks it back.
	a.SetFloor(0)
	if a.Target != 2 {
		t.Fatalf("Target after SetFloor(0) = %d, want 2 (unchanged)", a.Target)
	}
}

func TestAIMDConfigOverridesWatermarks(t *testing.T) {
	a := NewAIMD(AIMDConfig{KVHigh: 0.60, KVLow: 0.30})
	a.Target = 8
	sig := healthySignal(8)
	sig.KV = 0.65 // above the configured high watermark, below the default one
	if got := a.Update(sig); got != 4 {
		t.Fatalf("target = %d, want 4 under KVHigh 0.60", got)
	}
}

func TestEWMA(t *testing.T) {
	e := EWMA{Alpha: 0.5}
	if got := e.Observe(10); got != 10 {
		t.Fatalf("first sample = %v, want 10 (seeds the average)", got)
	}
	if got := e.Observe(0); got != 5 {
		t.Fatalf("second sample = %v, want 5", got)
	}
}

func TestEWMADefaultsToSpecAlpha(t *testing.T) {
	var e EWMA
	e.Observe(10)
	if got := e.Observe(0); got != 5 {
		t.Fatalf("second sample = %v, want 5 under the default alpha %v", got, EWMAAlpha)
	}
}

func TestTokenBucketRefills(t *testing.T) {
	t0 := time.Unix(0, 0)
	b := TokenBucket{Rate: 0.2, Capacity: 1}
	if !b.TryTake(t0) {
		t.Fatal("first take at t0 must succeed: the bucket starts full")
	}
	if b.TryTake(t0.Add(time.Second)) {
		t.Fatal("take at t0+1s must fail: only 0.2 tokens have refilled")
	}
	if !b.TryTake(t0.Add(5 * time.Second)) {
		t.Fatal("take at t0+5s must succeed: 1.0 token has refilled")
	}
}

func TestTokenBucketIgnoresBackwardsClock(t *testing.T) {
	t0 := time.Unix(0, 0)
	b := TokenBucket{Rate: 0.2, Capacity: 1}
	if !b.TryTake(t0) {
		t.Fatal("first take must succeed")
	}
	if b.TryTake(t0.Add(-time.Hour)) {
		t.Fatal("a backwards clock must mint no tokens")
	}
}

func TestTokenBucketCapsAtCapacity(t *testing.T) {
	t0 := time.Unix(0, 0)
	b := TokenBucket{Rate: 0.2, Capacity: 1}
	if !b.TryTake(t0) {
		t.Fatal("first take must succeed")
	}
	// An hour of refill is worth 720 tokens but the bucket holds one.
	if !b.TryTake(t0.Add(time.Hour)) {
		t.Fatal("take after a long idle must succeed")
	}
	if b.TryTake(t0.Add(time.Hour)) {
		t.Fatal("the bucket must not have banked more than its capacity")
	}
}
