package payments

import (
	"testing"

	"github.com/eigeninference/d-inference/coordinator/registry"
)

func TestLaneMultiplier(t *testing.T) {
	if got := LaneMultiplier(registry.LaneOnline); got != 1.0 {
		t.Errorf("LaneMultiplier(LaneOnline) = %v, want 1.0", got)
	}
	if got := LaneMultiplier(registry.LaneBatch); got != BatchDiscount {
		t.Errorf("LaneMultiplier(LaneBatch) = %v, want %v", got, BatchDiscount)
	}
	if BatchDiscount != 0.5 {
		t.Errorf("BatchDiscount = %v, want 0.5", BatchDiscount)
	}
}

func TestCalculateCostForLaneOnlineMatchesCalculateCostWithOverrides(t *testing.T) {
	// LaneOnline must be byte-for-byte identical to the existing (non-lane)
	// pricing path — the multiplier is 1.0, so nothing about online billing
	// changes.
	tests := []struct {
		name             string
		promptTokens     int
		completionTokens int
		customInput      int64
		customOutput     int64
		hasCustom        bool
	}{
		{"fallback, both zero", 0, 0, 0, 0, false},
		{"fallback, 1M+1M", 1_000_000, 1_000_000, 0, 0, false},
		{"fallback, tiny request under minimum", 10, 0, 0, 0, false},
		{"custom prices", 500_000, 500_000, 30_000, 165_000, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			want := CalculateCostWithOverrides("any-model", tt.promptTokens, tt.completionTokens, tt.customInput, tt.customOutput, tt.hasCustom)
			got := CalculateCostForLane("any-model", tt.promptTokens, tt.completionTokens, tt.customInput, tt.customOutput, tt.hasCustom, registry.LaneOnline)
			if got != want {
				t.Errorf("CalculateCostForLane(..., LaneOnline) = %d, want %d (CalculateCostWithOverrides)", got, want)
			}
		})
	}
}

func TestCalculateCostForLaneBatchHalvesFallbackPrice(t *testing.T) {
	// 1M prompt + 1M completion at fallback prices: online = 250,000 µUSD,
	// batch = 125,000 µUSD (docs/design/tidal-batch-lane.md §3.5).
	online := CalculateCostForLane("any-model", 1_000_000, 1_000_000, 0, 0, false, registry.LaneOnline)
	if online != 250_000 {
		t.Fatalf("online cost = %d, want 250000", online)
	}
	batch := CalculateCostForLane("any-model", 1_000_000, 1_000_000, 0, 0, false, registry.LaneBatch)
	if batch != 125_000 {
		t.Fatalf("batch cost = %d, want 125000", batch)
	}
	if batch != online/2 {
		t.Fatalf("batch cost %d is not half of online cost %d", batch, online)
	}
}

func TestCalculateCostForLaneBatchHasNoMinimumCharge(t *testing.T) {
	// A small batch request (10 completion tokens, fallback output price
	// $0.20/1M) rounds to 2 µUSD raw, 1 µUSD after the 0.5x batch discount.
	// The online path floors the same request at minimumChargeMicroUSD
	// (100 µUSD); the batch path must not apply that floor at all.
	const completionTokens = 10

	online := CalculateCostForLane("any-model", 0, completionTokens, 0, 0, false, registry.LaneOnline)
	if online != minimumChargeMicroUSD {
		t.Fatalf("online cost for a %d-token request = %d, want the %d minimum charge", completionTokens, online, minimumChargeMicroUSD)
	}

	batch := CalculateCostForLane("any-model", 0, completionTokens, 0, 0, false, registry.LaneBatch)
	if batch >= minimumChargeMicroUSD {
		t.Fatalf("batch cost = %d, want it under the %d minimum charge (no floor for LaneBatch)", batch, minimumChargeMicroUSD)
	}
	if batch != 1 {
		t.Fatalf("batch cost for a %d-token request = %d, want 1 (2 µUSD raw x 0.5 discount)", completionTokens, batch)
	}
}

func TestCalculateCostForLaneBatchNeverFree(t *testing.T) {
	// A nonzero token count that rounds to 0 µUSD after the discount is still
	// floored at 1 µUSD, matching CalculateCostWithOverridesNoMinimum's
	// anti-giveaway rule for other no-minimum channels.
	batch := CalculateCostForLane("any-model", 1, 0, 0, 0, false, registry.LaneBatch)
	if batch != 1 {
		t.Fatalf("batch cost for a 1 prompt-token request = %d, want 1 (never free for nonzero usage)", batch)
	}

	zero := CalculateCostForLane("any-model", 0, 0, 0, 0, false, registry.LaneBatch)
	if zero != 0 {
		t.Fatalf("batch cost for a zero-token request = %d, want 0", zero)
	}
}

func TestCalculateCostForLaneAppliesToServiceCustomPrices(t *testing.T) {
	// Custom (provider/platform) prices resolve exactly as before; the lane
	// multiplier is applied after that resolution, before rounding.
	const customInput, customOutput int64 = 30_000, 165_000
	online := CalculateCostForLane("any-model", 1_000_000, 1_000_000, customInput, customOutput, true, registry.LaneOnline)
	want := CalculateCostWithOverrides("any-model", 1_000_000, 1_000_000, customInput, customOutput, true)
	if online != want {
		t.Fatalf("online custom-price cost = %d, want %d", online, want)
	}
	batch := CalculateCostForLane("any-model", 1_000_000, 1_000_000, customInput, customOutput, true, registry.LaneBatch)
	if batch != want/2 {
		t.Fatalf("batch custom-price cost = %d, want %d (half of online)", batch, want/2)
	}
}

func TestProviderPayoutOnBatchCostAtZeroFeeEqualsDiscountedCost(t *testing.T) {
	batchCost := CalculateCostForLane("any-model", 1_000_000, 1_000_000, 0, 0, false, registry.LaneBatch)
	payout := ProviderPayoutWithPercent(batchCost, nil) // global default fee is 0% in this build
	if payout != batchCost {
		t.Fatalf("provider payout at 0%% fee = %d, want %d (the discounted cost)", payout, batchCost)
	}
}
