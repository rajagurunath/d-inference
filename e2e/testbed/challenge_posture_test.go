package testbed

import (
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/api"
)

// The default posture skips attestation challenges — what every e2e suite has
// always done. The coordinator keeps challenge freshness alive while skipping,
// so this stays safe for a suite that outlives the routing gate's 16-minute
// freshness window.
func TestChallengePostureDefaultsToSkipping(t *testing.T) {
	skip, interval := DefaultSuiteConfig().challengePosture()
	if !skip {
		t.Fatal("challenges should be skipped by default")
	}
	if interval != time.Hour {
		t.Fatalf("interval = %v, want the parked hour", interval)
	}
}

// RealChallenges takes the production challenge/response path on the
// coordinator's own default interval.
func TestChallengePostureRealChallengesUsesProductionInterval(t *testing.T) {
	skip, interval := SuiteConfig{RealChallenges: true}.challengePosture()
	if skip {
		t.Fatal("RealChallenges must not skip challenges")
	}
	if interval != api.DefaultChallengeInterval {
		t.Fatalf("interval = %v, want api.DefaultChallengeInterval (%v)", interval, api.DefaultChallengeInterval)
	}
	if interval > api.MaxSkippedChallengeRefreshInterval {
		t.Fatalf("interval %v would leave the freshness gate unrefreshed between challenges", interval)
	}
}
