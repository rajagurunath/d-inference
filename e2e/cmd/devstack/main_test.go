package main

import "testing"

// Persistence resolution lives here rather than in testbed because
// EIGENINFERENCE_DATABASE_URL is a name the testbed writes to itself
// (deps.PostgresLifecycle.SetEnv). These tests pin the precedence and, more
// importantly, that neither variable can turn a dev loop that did not ask for
// Postgres into one.

func TestResolveDatabaseURLPrefersTheDevstackVariable(t *testing.T) {
	t.Setenv(devstackDatabaseURLEnv, "postgres://devstack/db")
	t.Setenv("EIGENINFERENCE_DATABASE_URL", "postgres://coordinator/db")

	url, source := resolveDatabaseURL(true)
	if url != "postgres://devstack/db" || source != devstackDatabaseURLEnv {
		t.Fatalf("resolveDatabaseURL = (%q, %q), want the devstack variable", url, source)
	}
}

func TestResolveDatabaseURLFallsBackToTheCoordinatorVariable(t *testing.T) {
	t.Setenv(devstackDatabaseURLEnv, "")
	t.Setenv("EIGENINFERENCE_DATABASE_URL", "postgres://coordinator/db")

	url, source := resolveDatabaseURL(true)
	if url != "postgres://coordinator/db" || source != "EIGENINFERENCE_DATABASE_URL" {
		t.Fatalf("resolveDatabaseURL = (%q, %q), want the coordinator variable", url, source)
	}
}

// Without --postgres neither variable is read, so an exported URL cannot
// silently move the dev loop off the in-memory store.
func TestResolveDatabaseURLIgnoresBothWithoutPostgres(t *testing.T) {
	t.Setenv(devstackDatabaseURLEnv, "postgres://devstack/db")
	t.Setenv("EIGENINFERENCE_DATABASE_URL", "postgres://coordinator/db")

	if url, source := resolveDatabaseURL(false); url != "" || source != "" {
		t.Fatalf("resolveDatabaseURL(false) = (%q, %q), want empty", url, source)
	}
}

// --postgres with neither variable set keeps the ephemeral instance the stack
// provisions and drops.
func TestResolveDatabaseURLEmptyMeansEphemeral(t *testing.T) {
	t.Setenv(devstackDatabaseURLEnv, "")
	t.Setenv("EIGENINFERENCE_DATABASE_URL", "")

	if url, source := resolveDatabaseURL(true); url != "" || source != "" {
		t.Fatalf("resolveDatabaseURL = (%q, %q), want empty", url, source)
	}
}
