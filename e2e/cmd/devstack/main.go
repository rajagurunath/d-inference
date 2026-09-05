// Command devstack runs a coordinator and one provider on fixed, well-known
// addresses for manual testing on a single Mac. It is the same in-process
// testbed.Suite the e2e integration tests use — never a hand-rolled
// coordinator or provider launch — so the dev loop exercises exactly the
// lifecycle CI does.
//
// Usage:
//
//	cd e2e && GOTOOLCHAIN=auto go run ./cmd/devstack [flags]
//
// The coordinator listens on http://127.0.0.1:18080. Stop with Ctrl-C
// (SIGINT); the suite tears down the provider and coordinator cleanly.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/eigeninference/d-inference/e2e/testbed"
)

const devstackListenAddr = "127.0.0.1:18080"

func main() {
	usePostgres := flag.Bool("postgres", false, "use a real Postgres store instead of the in-memory store: an ephemeral instance the stack provisions and drops on exit, or — when DARKBLOOM_DEVSTACK_DATABASE_URL (or, failing that, EIGENINFERENCE_DATABASE_URL) names one — that database, kept across restarts")
	model := flag.String("model", "", "model ID to serve (default: DARKBLOOM_TESTBED_MODEL env, then the testbed default)")
	providerBinary := flag.String("provider-binary", "", "path to a prebuilt provider binary (default: DARKBLOOM_PROVIDER_BINARY env, else build from source via testbed.BuildProvider)")
	apiKey := flag.String("api-key", os.Getenv("DARKBLOOM_DEV_KEY"), "reuse this already-issued key for the printed user instead of minting one (default: DARKBLOOM_DEV_KEY env). Only a key the store already holds can be reused — see the WARN it logs otherwise")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	if *providerBinary != "" {
		if err := os.Setenv("DARKBLOOM_PROVIDER_BINARY", *providerBinary); err != nil {
			logger.Error("set DARKBLOOM_PROVIDER_BINARY", "error", err)
			os.Exit(1)
		}
	}

	cfg := testbed.SuiteConfig{
		UseMemoryStore: !*usePostgres,
		ListenAddr:     devstackListenAddr,
		APIKey:         *apiKey,
	}
	if *model != "" {
		cfg.ModelSpecs = []testbed.ModelSpec{{ModelID: *model, NumProviders: 1}}
	}

	// Persistence is resolved HERE, not in testbed: the suite honours only an
	// explicit SuiteConfig.DatabaseURL, because the runner-up variable is one
	// the testbed writes to itself (deps.PostgresLifecycle.SetEnv exports an
	// ephemeral instance's URL), and a suite reading it back would attach to a
	// previous suite's removed database.
	databaseURL, urlSource := resolveDatabaseURL(*usePostgres)
	cfg.DatabaseURL = databaseURL
	persistentDB := databaseURL != ""
	if persistentDB {
		logger.Info("persistent postgres store", "source", urlSource)
	}

	// Restart resilience needs all three of: a database that outlives the
	// process, the key that addresses its rows, and a seal key that can still
	// open the blobs the previous process wrote. The first two are above; this
	// is the third. A process-local random key
	// (EIGENINFERENCE_BATCH_DEV_INSECURE_KEY) makes every sealed batch input
	// undecryptable at restart, so when the developer has asked for a
	// persistent database and named no mnemonic, pin the fixed dev one and
	// take the production derivation path instead.
	if persistentDB && testbed.ResolveEncryptionMnemonic("") == "" {
		cfg.EncryptionMnemonic = testbed.DevRestartMnemonic
		logger.Warn("batch lane sealing with the FIXED DEV MNEMONIC so blobs survive a restart — " +
			"it is a publicly known BIP39 test vector, never a secret. Set MNEMONIC to use your own; " +
			"never run a deployment on this value.")
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	suite := testbed.NewSuite(cfg)
	logger.Info("starting dev stack",
		"listen_addr", devstackListenAddr,
		"postgres", *usePostgres,
		"persistent_db", persistentDB,
		"model", cfg.PrimaryModelID())
	if err := suite.Start(ctx); err != nil {
		logger.Error("dev stack failed to start", "error", err)
		os.Exit(1)
	}

	fmt.Printf("\nDev stack ready.\n")
	fmt.Printf("  Base URL: %s\n", suite.Coordinator.BaseURL())
	if len(suite.Users) > 0 {
		fmt.Printf("  API key:  %s\n", suite.Users[0].APIKey)
	}
	for _, p := range suite.Providers {
		fmt.Printf("  Provider PID: %d (model: %s)\n", p.PID(), cfg.PrimaryModelID())
	}
	if persistentDB {
		fmt.Printf("\nPersistent store (%s): this database is kept on exit. Restart with\n", urlSource)
		fmt.Printf("  DARKBLOOM_DEV_KEY=%s make dev-stack   # same DARKBLOOM_DEVSTACK_DATABASE_URL and blob dir\n",
			devUserKey(suite))
		fmt.Printf("to resume open batches with the same key.\n")
	}
	fmt.Printf("\nTo run the smoke script:\n")
	if len(suite.Users) > 0 {
		fmt.Printf("  DARKBLOOM_DEV_URL=%s DARKBLOOM_DEV_KEY=%s ./scripts/dev-smoke-chat.sh\n", suite.Coordinator.BaseURL(), suite.Users[0].APIKey)
	}
	fmt.Printf("\nPress Ctrl-C to stop.\n\n")

	<-ctx.Done()
	logger.Info("shutting down dev stack")
	suite.Stop()
	logger.Info("dev stack stopped")
}

// devstackDatabaseURLEnv is the devstack's own persistence variable. It is
// deliberately not EIGENINFERENCE_DATABASE_URL: that name is also written by
// the testbed's ephemeral Postgres lifecycle (deps.PostgresLifecycle.SetEnv),
// so it cannot be trusted as a statement of developer intent inside a process
// that provisions databases of its own.
const devstackDatabaseURLEnv = "DARKBLOOM_DEVSTACK_DATABASE_URL"

// resolveDatabaseURL returns the database the dev stack should keep across
// restarts, and the name of the variable it came from.
//
// Both variables are read only under --postgres, so neither can silently turn
// the in-memory dev loop into a Postgres one. EIGENINFERENCE_DATABASE_URL is
// the second choice, kept because it is the name the flag help has always
// carried and the one a developer already has exported for the coordinator.
func resolveDatabaseURL(usePostgres bool) (url, source string) {
	if !usePostgres {
		return "", ""
	}
	if u := os.Getenv(devstackDatabaseURLEnv); u != "" {
		return u, devstackDatabaseURLEnv
	}
	if u := os.Getenv("EIGENINFERENCE_DATABASE_URL"); u != "" {
		return u, "EIGENINFERENCE_DATABASE_URL"
	}
	return "", ""
}

// devUserKey is the raw key for the printed user, or a placeholder when the
// suite created none.
func devUserKey(suite *testbed.Suite) string {
	if len(suite.Users) == 0 {
		return "<no user>"
	}
	return suite.Users[0].APIKey
}
