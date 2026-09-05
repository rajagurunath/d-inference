package registry

import (
	"testing"
	"time"
)

// challenge_freshness_skip_test.go — the routing liveness gate's challenge
// freshness window and RefreshChallengeFreshnessSkipped, the restamp a
// skipChallenge coordinator uses instead of a verified challenge.

const skipFreshnessModel = "mlx-community/Qwen3.5-9B-Instruct-4bit"

// livenessGateReason evaluates the shared liveness core under the correct lock
// discipline and returns the first closed gate (GateReasonCount = open).
func livenessGateReason(reg *Registry, p *Provider) GateReason {
	reg.mu.RLock()
	defer reg.mu.RUnlock()
	p.mu.Lock()
	defer p.mu.Unlock()
	_, reason := reg.providerLivenessGateReasonLocked(p, reg.MinTrustLevel, false, time.Now())
	return reason
}

// TestChallengeFreshnessGateClosesWithoutRefresh pins the defect the skip-
// challenge refresh loop exists to avoid: a provider whose stamp is written
// once (at registration) and never refreshed is deroutable once the stamp ages
// past challengeFreshnessMaxAge, even though it is connected, online and
// serving the model. If this test ever fails, the gate has changed and the
// refresh loop in coordinator/api may no longer be needed.
func TestChallengeFreshnessGateClosesWithoutRefresh(t *testing.T) {
	reg := New(testLogger())
	p := reg.Register("p-skip-stale", nil, testRegisterMessage())
	reg.SetTrustLevel(p.ID, TrustHardware)
	reg.RecordChallengeSuccess(p.ID)

	if findRoutableProvider(reg, skipFreshnessModel) == nil {
		t.Fatal("baseline: a freshly stamped provider must be routable")
	}

	// Age the registration stamp past the window — the state a skipChallenge
	// coordinator reaches ~16 minutes after the provider connects.
	p.SetLastChallengeVerified(time.Now().Add(-challengeFreshnessMaxAge - time.Minute))

	if got := livenessGateReason(reg, p); got != GateChallengeStale {
		t.Fatalf("liveness gate = %v, want %v", got, GateChallengeStale)
	}
	if findRoutableProvider(reg, skipFreshnessModel) != nil {
		t.Fatal("a provider with a stale challenge stamp must not be routable")
	}
	if len(reg.ModelCapacitySnapshot()) != 0 {
		t.Fatal("capacity feed must be empty while the challenge stamp is stale")
	}
}

// TestRefreshChallengeFreshnessSkippedKeepsProviderRoutable is the regression
// for the idle dropout: with challenges skipped, the periodic restamp is the
// only thing holding the gate open, and it must reopen it.
func TestRefreshChallengeFreshnessSkippedKeepsProviderRoutable(t *testing.T) {
	reg := New(testLogger())
	p := reg.Register("p-skip-refresh", nil, testRegisterMessage())
	reg.SetTrustLevel(p.ID, TrustHardware)
	reg.RecordChallengeSuccess(p.ID)

	stale := time.Now().Add(-challengeFreshnessMaxAge - time.Minute)
	p.SetLastChallengeVerified(stale)
	if findRoutableProvider(reg, skipFreshnessModel) != nil {
		t.Fatal("precondition: the stale provider should have been derouted")
	}

	reg.RefreshChallengeFreshnessSkipped(p.ID)

	if !p.GetLastChallengeVerified().After(stale) {
		t.Fatalf("LastChallengeVerified = %v, want a fresh stamp", p.GetLastChallengeVerified())
	}
	if got := livenessGateReason(reg, p); got != GateReasonCount {
		t.Fatalf("liveness gate = %v, want open", got)
	}
	if findRoutableProvider(reg, skipFreshnessModel) == nil {
		t.Fatal("a restamped provider must be routable again")
	}
	if len(reg.ModelCapacitySnapshot()) == 0 {
		t.Fatal("capacity feed must list the restamped provider's model")
	}
}

// TestRefreshChallengeFreshnessSkippedProvesNothing keeps the method honest:
// it moves the freshness clock and nothing else. Unlike RecordChallengeSuccess
// it must not claim a SIP verification or clear a failed-challenge count — no
// attestation was performed.
func TestRefreshChallengeFreshnessSkippedProvesNothing(t *testing.T) {
	reg := New(testLogger())
	p := reg.Register("p-skip-honest", nil, testRegisterMessage())

	p.mu.Lock()
	p.ChallengeVerifiedSIP = false
	p.FailedChallenges = 2
	p.mu.Unlock()

	reg.RefreshChallengeFreshnessSkipped(p.ID)

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.LastChallengeVerified.IsZero() {
		t.Fatal("LastChallengeVerified should have been stamped")
	}
	if p.ChallengeVerifiedSIP {
		t.Error("ChallengeVerifiedSIP must stay false: nothing was verified")
	}
	if p.FailedChallenges != 2 {
		t.Errorf("FailedChallenges = %d, want 2 (unchanged)", p.FailedChallenges)
	}
}

// An unknown provider ID is a no-op: the loop races Disconnect.
func TestRefreshChallengeFreshnessSkippedUnknownProvider(t *testing.T) {
	reg := New(testLogger())
	reg.RefreshChallengeFreshnessSkipped("nobody")
}

// ChallengeFreshnessMaxAge exposes the gate's own constant; a second copy of
// the number in coordinator/api is exactly what this accessor prevents.
func TestChallengeFreshnessMaxAgeMatchesGate(t *testing.T) {
	if ChallengeFreshnessMaxAge() != challengeFreshnessMaxAge {
		t.Fatalf("ChallengeFreshnessMaxAge() = %v, want %v", ChallengeFreshnessMaxAge(), challengeFreshnessMaxAge)
	}
}
